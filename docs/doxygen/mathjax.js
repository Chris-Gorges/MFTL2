/*
 * MathJax 3 configuration for the MFTL manual.
 *
 * Doxygen injects this file's contents into the MathJax bootstrap, so it runs
 * BEFORE MathJax starts and may set MathJax.config fields directly.
 *
 * Two things are configured beyond the defaults:
 *
 *   - `$ ... $` is enabled as an inline delimiter. Doxygen emits `\f$ ... \f$`
 *     as `\( ... \)`, which MathJax already understands, but the Markdown pages
 *     (the module READMEs) are also read on GitHub, where `$ ... $` is the
 *     dialect that renders. Accepting both means one source serves both readers.
 *   - `\[ ... \]` and `$$ ... $$` are both display delimiters, for the same
 *     reason.
 *
 * `processEscapes` keeps a literal `\$` out of the parser, which matters because
 * several pages quote shell snippets.
 */
window.MathJax = {
  tex: {
    inlineMath: [
      ['\\(', '\\)'],
      ['$', '$']
    ],
    displayMath: [
      ['\\[', '\\]'],
      ['$$', '$$']
    ],
    processEscapes: true,
    processEnvironments: true,
    packages: { '[+]': ['ams', 'boldsymbol'] },
    macros: {
      /* Shorthands used across the theory pages. Keep them in step with
       * docs/doxygen/formulas.tex, which serves the LaTeX/PDF backend. */
      vect: ['\\mathbf{#1}', 1],
      norm: ['\\left\\lVert #1 \\right\\rVert', 1],
      abs: ['\\left\\lvert #1 \\right\\rvert', 1],
      set: ['\\left\\{ #1 \\right\\}', 1],
      nlocal: '\\ensuremath{n_{\\mathrm{local}}}',
      orient: '\\operatorname{orient3d}',
      sign: '\\operatorname{sign}',
      clz: '\\operatorname{clz}'
    }
  },
  options: {
    /* Doxygen wraps code in <code>/<pre>; never typeset inside those. */
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'annotation', 'annotation-xml'],
    ignoreHtmlClass: 'fragment|line|memname'
  },
  chtml: {
    scale: 1.0,
    matchFontHeight: true
  },
  startup: {
    typeset: true
  }
};
