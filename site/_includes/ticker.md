```{=html}
<!--
  Live quotes from TradingView's free ticker widgets. They supply and license the
  data; nothing here comes from this project's own price files.

  The scrolling tape is loaded only for visitors who have not asked for reduced
  motion. They get TradingView's static ticker row instead. Without JavaScript the
  band stays empty and the rest of the page is unaffected.
-->
<div class="ticker-band">
  <div class="tradingview-widget-container" id="tv-ticker">
    <div class="tradingview-widget-container__widget"></div>
  </div>
  <p class="ticker-credit"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">Quotes by TradingView</a></p>
</div>
<script>
  (function () {
    var holder = document.getElementById("tv-ticker");
    if (!holder) return;
    var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var symbols = [
      { proName: "NASDAQ:NVDA", title: "NVIDIA" },
      { proName: "NASDAQ:AMD", title: "AMD" },
      { proName: "NYSE:TSM", title: "TSMC" },
      { proName: "NASDAQ:AVGO", title: "Broadcom" },
      { proName: "NASDAQ:SMH", title: "Semiconductor ETF" },
      { proName: "NASDAQ:QQQ", title: "Nasdaq 100 ETF" },
      { proName: "AMEX:SPY", title: "S&P 500 ETF" }
    ];
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.async = true;
    script.src =
      "https://s3.tradingview.com/external-embedding/" +
      (still ? "embed-widget-tickers.js" : "embed-widget-ticker-tape.js");
    script.text = JSON.stringify({
      symbols: still ? symbols.slice(0, 4) : symbols,
      colorTheme: "dark",
      isTransparent: true,
      showSymbolLogo: true,
      displayMode: "regular",
      locale: "en"
    });
    holder.appendChild(script);
  })();
</script>
```
