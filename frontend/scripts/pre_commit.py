#!/usr/bin/env python3
import subprocess
import sys

def main():
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        text=True
    ).strip()

    if branch in ["main", "develop"]:
        print(
            f"Commit direto para '{branch}' não permitido! "
            "Crie uma branch de feature a partir da develop antes de commitar."
        )
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
