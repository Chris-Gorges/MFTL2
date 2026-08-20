/*
 * MathJax 3 configuration for the MFTL manual (MkDocs / Material).
 *
 * Arithmatex runs in "generic" mode, so it hands MathJax `\( ... \)` and
 * `\[ ... \]` rather than raw dollars. Both delimiter pairs are still declared
 * because two of the three content sources reach this page unmodified:
 *
 *   - the module READMEs carry GitHub-dialect `$ ... $`, because they are read
 *     on GitHub too;
 *   - mkdoxy emits `$ ... $` for every `<formula>` element it finds in the
 *     Doxygen XML, which is what the `\f$ ... \f$` in the headers becomes.
 *
 * The macro list is the same one docs/doxygen/formulas.tex and
 * docs/doxygen/mathjax.js define for the Doxygen manual. Keep the three in
 * step: a missing macro renders as a red \vect rather than as an error.
 */
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"]
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"]
    ],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams", "boldsymbol"] },
    macros: {
      vect: ["\\mathbf{#1}", 1],
      norm: ["\\left\\lVert #1 \\right\\rVert", 1],
      abs: ["\\left\\lvert #1 \\right\\rvert", 1],
      set: ["\\left\\{ #1 \\right\\}", 1],
      nlocal: "n_{\\mathrm{local}}",
      orient: "\\operatorname{orient3d}",
      sign: "\\operatorname{sign}",
      clz: "\\operatorname{clz}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code", "annotation", "annotation-xml"]
  }
};

/*
 * Material's instant loading swaps the document body without a page load, so
 * MathJax has to be told to typeset again. `document$` is the observable
 * Material exposes for exactly this.
 */
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
