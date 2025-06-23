document.addEventListener("DOMContentLoaded", function() {
  if (typeof katex === "undefined") return;
  const delimiters = [
    { left: "\\(", right: "\\)", displayMode: false },
    { left: "\\[", right: "\\]", displayMode: true },
  ]
  const elements = document.querySelectorAll('.arithmatex');
  elements.forEach(el => {
    const matched = delimiters.find(d => rawTex.startsWith(d.left) && rawTex.endsWith(d.right));
    if (!matched) return;
    const tex = rawTex.slice(matched.left.length, rawTex.length - matched.right.length);
    try {
      katex.render(tex, el, { displayMode: matched.displayMode });
    } catch (err) {
      console.error("KaTeX rendering error:", err.message);
      el.innerHTML = '<span style="color: red;">Error rendering LaTeX</span>';
    }
  })
});