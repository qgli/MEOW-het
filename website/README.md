# MEOW project page

Static project website, published at <https://qgli.github.io/MEOW-het/>.

Edit `index.html` for content and links, `style.css` for layout, and `app.js` for
the small browser-side interactions. Preview from the repository root:

```sh
python3 -m http.server 18788 --bind 127.0.0.1 --directory website
```

Push changes to `main`; `.github/workflows/pages.yml` publishes this directory
with GitHub Pages. No build tools, API keys or server are needed. The site links
to the live Hugging Face demo rather than loading the inference application in
the background. Opening this page consumes no model inference quota.

Research figures and the orbit animation come from the released MEOW repository.
The four example thumbnails come from the demo's privacy-redacted cached example.
Benchmark numbers follow the public repository's per-tuple aggregation.
