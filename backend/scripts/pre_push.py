#!/usr/bin/env python3
import subprocess
import sys

def main():
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        text=True
    ).strip()

    if branch in ["main", "develop"]:
        print(f"Nao é permitido push direto para a branch '{branch}'!"
         "Crie uma branch para commitar suas mudancas e abra Pull Request no GitHub."
        )
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
