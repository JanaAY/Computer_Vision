# Publishing to your GitHub account

The repository files are ready locally, but no repository has been created in your GitHub account by this package. To publish:

1. On GitHub, create an **empty** repository named `inspectai-pcb-vision` (do not add another README or .gitignore).
2. In this directory run:

```bash
git init
git add .
git commit -m "Initial InspectAI architecture, roadmap and CPU baseline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/inspectai-pcb-vision.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. For the GitHub CLI alternative, authenticate using `gh auth login` then run `gh repo create inspectai-pcb-vision --public --source=. --remote=origin --push` **after** initializing/committing locally. Never commit datasets, model weights you lack rights to redistribute, API keys, or confidential production images.
