# harsh-personal-website

Personal site and blog for [Harsh Thomare](https://harshthomare.github.io/).

**Live site:** [harshthomare.github.io](https://harshthomare.github.io/)

GitHub Pages serves that root URL only from a repository named `HarshThomare.github.io`. This repository has to be renamed to that before the workflow publishes there. The deploy build already takes its base URL from Pages (`steps.pages.outputs.base_url`), so the workflow does not need a hardcoded path.

Built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Local development

```bash
git clone --recurse-submodules https://github.com/HarshThomare/harsh-personal-website.git
cd harsh-personal-website
hugo server
```

Open [http://localhost:1313](http://localhost:1313).

## License

Site content © Harsh Thomare. PaperMod theme is [MIT licensed](https://github.com/adityatelange/hugo-PaperMod/blob/master/LICENSE).
