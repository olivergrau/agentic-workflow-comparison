import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, "./")
output_file = os.path.join(base_dir, "test_results.txt")

# List of test script filenames
test_scripts = [
    "action_planning_agent.py",
    "augmented_prompt_agent.py",
    "direct_prompt_agent.py",
    "evaluation_agent.py",
    "knowledge_augmented_prompt_agent.py",
    "rag_knowledge_prompt_agent.py",
    "routing_agent.py"
]

# Run each script and capture output
with open(output_file, "w", encoding="utf-8") as out_file:
    for script in test_scripts:
        script_path = os.path.join(test_dir, script)
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            continue

        header = f"\n{'='*40}\nRunning: {script}\n{'='*40}\n"
        print(header)
        out_file.write(header)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                cwd=test_dir,
                check=False  # don't raise exceptions on non-zero exit
            )

            print(result.stdout)
            out_file.write(result.stdout)

            if result.stderr:
                print("⚠️ stderr:")
                print(result.stderr)
                out_file.write("\n[stderr]\n" + result.stderr)

        except Exception as e:
            error_msg = f"\n[ERROR] Could not run {script}: {e}\n"
            print(error_msg)
            out_file.write(error_msg)

print(f"\n✅ All tests completed. Results written to {output_file}")
