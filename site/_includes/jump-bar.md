```{=html}
<script>
  // The NVIDIA page's jump bar.
  //
  // 1. Keeps the bar just under the site header. The header hides when the page
  //    scrolls down and comes back when it scrolls up (Quarto's "headroom"), so
  //    the bar sits at the very top while the header is hidden and below it while
  //    it is shown.
  // 2. Handles the bar's own links. Quarto already leaves room for its header on a
  //    jump, in two different ways; added to the bar that over-counts, so these
  //    links scroll themselves: the section's eyebrow line lands below the header
  //    and the bar together. Whether the header hides after a jump cannot be
  //    relied on (measured: after an instant jump down it stayed), so room is
  //    always left for both. Focus moves to the heading, so keyboard users
  //    continue from there. Without JavaScript the links still work as ordinary
  //    anchors.
  (function () {
    var bar = document.querySelector(".jump-bar");
    if (!bar) return;
    var header = document.getElementById("quarto-header");
    var gap = 12;

    function headerHeight() {
      return header ? header.offsetHeight : 0;
    }

    function place() {
      var shown = header && !header.classList.contains("headroom--unpinned");
      bar.style.top = (shown ? headerHeight() : 0) + "px";
      document.documentElement.style.setProperty("--jump-bar-height", bar.offsetHeight + "px");
    }

    place();
    window.addEventListener("resize", place);
    if (header && "MutationObserver" in window) {
      new MutationObserver(place).observe(header, {
        attributes: true,
        attributeFilter: ["class"]
      });
    }

    var still = window.matchMedia("(prefers-reduced-motion: reduce)");

    bar.addEventListener("click", function (event) {
      var link = event.target.closest("a");
      if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var id = new URL(link.href).hash.slice(1);
      var section = id && document.getElementById(id);
      if (!section) return;
      event.preventDefault();

      // Land on the eyebrow line above the heading when there is one.
      var before = section.previousElementSibling;
      var anchor = before && before.querySelector(".eyebrow") ? before : section;
      var top = anchor.getBoundingClientRect().top + window.scrollY;
      var covered = bar.offsetHeight + headerHeight();

      history.pushState(null, "", "#" + id);
      window.scrollTo({ top: Math.max(top - covered - gap, 0), behavior: still.matches ? "auto" : "smooth" });
      var heading = section.querySelector("h2") || section;
      if (!heading.hasAttribute("tabindex")) heading.setAttribute("tabindex", "-1");
      heading.focus({ preventScroll: true });
    });
  })();
</script>
```
