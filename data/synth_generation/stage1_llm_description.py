import time
import httpx
import json
import re
import os
import sys
from openai import OpenAI

# ==========================================
# 1. Configuration
# ==========================================
MY_API_KEY = "YOUR_API_KEY_HERE"
MY_BASE_URL = "YOUR_BASE_URL_HERE"
MODEL_LIST = [
    "claude-sonnet-4-5-20250929",
]

# Save settings
SAVE_DIR = "./prompt.json"
LOG_ERRORS = True

anomaly_labels = [
    "Violent Conflict", "Crime", "Traffic Accident", "Personal Emergency", "Environmental Incident", "Public Misconduct"
]
GROUPS_PER_ROUND = 15
TOTAL_ROUNDS_PER_LABEL = 2
START_INDEX_OFFSET = 1

# Performance and circuit-breaker settings
REQUEST_TIMEOUT = 400.0
TOTAL_FAILURE_LIMIT = 3
COOLDOWN_TIME = 4

# ==========================================
# 2. Templates
# ==========================================
SYSTEM_TEMPLATE = """"""
USER_TEMPLATE = """"""

VAD_TASK_PROMPT = """
Role: You are a prompt generation expert for surveillance video anomaly detection datasets. The generated prompts will be fed into a text-to-image model, so semantic clarity is the top priority over fine-grained detail.

Task: Generate four-frame storyboard scene pairs for the category "{label}".

Requirements:
- CURRENT_LABEL={label}
- Generate {groups} differentiated pairs starting from index {start_idx:02d}; ensure sufficient variety between pairs
- Do not involve children
- **Use soft descriptions for anomalous semantics, but reinforce the anomaly semantics of {label}**

Generation Rules:
1. Design a [Shared Scene Pc]: specific location, time, and environmental details
2. Design [Characters Pchar]: number of people, appearance, clothing (must remain consistent across all four frames)
3. Design [Abnormal Four Frames]: follow the structure F1_pre → F2_start → F3_peak → F4_post
4. Design [Normal Four Frames]: show normal behavior under the exact same scene and character settings

Frame Description Requirements:
**F1_pre (Pre-event)**:
- Establish the scene baseline; characters are in a neutral, natural state
- Normal and abnormal branches should be [nearly identical] in this frame
- Key objects must be clearly visible in the frame
- Environment is clean and orderly

**F2_start (Action onset)**:
- Behavioral intent begins to emerge
- Abnormal branch: multiple visual warning signals or actions appear, conveying a strong anomalous foreground
- Normal branch: natural start of a routine action, relaxed posture

**F3_peak (Action peak)**: [Most critical frame — determines whether the anomaly is recognizable]
- Visual differences must be maximized
- Abnormal branch must include at least 3 specific, visible anomalous visual elements
- Description must be most detailed: character posture + object interaction + spatial relationships + facial/body language
- Normal branch: equally detailed description of corresponding elements in a normal state

**F4_post (Post-event)**:
- Show the consequences of the behavior
- Abnormal branch: environment/character state has visibly changed, leaving visual evidence
- Normal branch: scene remains or returns to a normal, orderly state

Consistency Requirements:
- The same character appears across all four frames with unchanged clothing and appearance
- Scene background is completely consistent across all four frames
- Clear temporal/causal progression between frames

After generation, confirm:
1. Can the abnormal/normal descriptions in F2_start and F3_peak be clearly distinguished by visual inspection alone?
2. Does the abnormal F3 contain at least 3 specific, visible anomalous visual elements?
3. Are the character clothing descriptions in all four frames fully consistent with Pchar?
4. Have prohibited terms been avoided and softened alternatives used?
5. Is the scene sufficiently different from previously generated pairs?

Output Format (JSON):
**Place all {groups} pairs in a single JSON array. Do not use markdown code blocks:**
[
{{
  "pair_id": "{label}_{{index}}",
  "shared_context": {{
    "Pc": "Specific scene description",
    "Pchar": "Character description (appearance + clothing)"
  }},
  "abnormal_branch": {{
    "label": "{label}",
    "Po": "Key objects",
    "frames": {{
      "F1_pre": "Abnormal pre-event description",
      "F2_start": "Abnormal action onset description",
      "F3_peak": "Abnormal action peak description",
      "F4_post": "Abnormal post-event description"
    }}
  }},
  "normal_branch": {{
    "label": "Normal",
    "Po": "Normal objects",
    "frames": {{
      "F1_pre": "Normal pre-event description",
      "F2_start": "Normal action onset description",
      "F3_peak": "Normal action peak description",
      "F4_post": "Normal post-event description"
    }}
  }}
}}
]
"""

# ==========================================
# 3. Core Logic
# ==========================================

client = OpenAI(
    api_key=MY_API_KEY,
    base_url=MY_BASE_URL,
    timeout=httpx.Timeout(connect=10.0, read=REQUEST_TIMEOUT, write=10.0, pool=None)
)


class VADGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def clean_and_extract(self, text, label, idx):
        # Strip <Think> tags and redundant content
        cleaned = re.sub(r'<Think>.*?</Think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'', '', cleaned)
        try:
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            if LOG_ERRORS:
                err_file = os.path.join(self.save_path, f"error_{label}_{idx}.txt")
                with open(err_file, "w", encoding="utf-8") as f:
                    f.write(text)
            print(f"\n[Parse Failed] Unable to extract JSON.")
        return None

    def call_api_with_fallback(self, label, start_idx):
        """
        API call with checkpoint resume and model fallback.
        """
        # Checkpoint check: skip if output file already exists
        save_filename = f"{label}_{start_idx:02d}.json"
        save_file = os.path.join(self.save_path, save_filename)

        if os.path.exists(save_file):
            print(f"⏩ [Skip] File already exists: {save_filename}")
            return True

        task_content = VAD_TASK_PROMPT.format(
            label=label, groups=GROUPS_PER_ROUND, start_idx=start_idx
        )

        for model in MODEL_LIST:
            try:
                print(f"\n>>> Trying model: {model} | Label: {label} | Index: {start_idx:02d}")

                messages = [
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": f"{task_content}\n\n{USER_TEMPLATE}"}
                ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )

                full_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        char = chunk.choices[0].delta.content
                        print(char, end="", flush=True)
                        full_text += char

                data = self.clean_and_extract(full_text, label, start_idx)
                if data:
                    save_file = os.path.join(self.save_path, f"{label}_{start_idx:02d}.json")
                    with open(save_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"\n✅ Success! Saved to: {save_file}")
                    return True

                print(f"\n[Info] Model {model} returned unparseable output. Trying next model...")

            except Exception as e:
                print(f"\n[API Error] Model {model} raised an error: {e}")
                time.sleep(2)

        # All models exhausted without success
        return False


# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    gen = VADGenerator(SAVE_DIR)
    total_failure_count = 0

    for label in anomaly_labels:
        for r in range(TOTAL_ROUNDS_PER_LABEL):
            current_idx = (r * GROUPS_PER_ROUND) + START_INDEX_OFFSET

            success = gen.call_api_with_fallback(label, current_idx)

            if success:
                time.sleep(COOLDOWN_TIME)
            else:
                total_failure_count += 1
                print(
                    f"\n[Warning] Label '{label}' round {r + 1} failed across all models. "
                    f"(Cumulative failures: {total_failure_count}/{TOTAL_FAILURE_LIMIT})"
                )

                if total_failure_count >= TOTAL_FAILURE_LIMIT:
                    print("\n[Circuit Breaker] Cumulative failure limit reached. Exiting to protect quota or investigate configuration.")
                    sys.exit(1)

    print("\n[Done] All tasks completed.")