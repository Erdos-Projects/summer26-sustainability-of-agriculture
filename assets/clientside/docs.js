/* The docs overlay: open/close, document switching, and link interception.
 *
 * Three clientside callbacks registered from widget/components/docs_panel.py, plus two delegated
 * listeners installed at load. The listeners are here rather than in a callback because they must
 * survive every re-render of the document body, and because a callback that installs a listener has
 * to guard against running twice; a script that runs once does not.
 *
 * The dcc.Markdown component is CONSTRUCTED HERE rather than sitting in the Python layout. With
 * mathjax=True in the layout, Dash pulls the 2 MB async-mathjax chunk at page boot -- on a page
 * whose point is a map. Built here, markdown loads on first open and MathJax only when a document
 * that contains math is selected.
 */

(function () {
    window.dash_clientside = window.dash_clientside || {};
    const NO = () => window.dash_clientside.no_update;
    const B = () => window.dash_clientside.bundle;
    const trig = () => window.dash_clientside.bundle.triggeredId();

    const NAV_PREFIX = "docs-nav-";

    // Stashed by render() so the delegated click handler -- which lives outside the callback graph
    // and so cannot take a Store as State -- can resolve relative hrefs. A cache of Store data, not
    // a second definition of it: docs_panel.payload() remains the source.
    let payloadCache = null;

    /* GitHub's heading-anchor slug, closely enough for the links these documents actually carry.
     * dcc.Markdown emits no heading ids, so an in-page anchor is matched on heading TEXT instead. */
    const slugify = (s) => s.toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");

    function scrollToHeading(frag) {
        const body = document.getElementById("docs-body");
        if (!body) return;
        const want = slugify(decodeURIComponent(frag));
        const hs = body.querySelectorAll("h1, h2, h3, h4");
        for (let i = 0; i < hs.length; i++) {
            if (slugify(hs[i].textContent) === want) {
                hs[i].scrollIntoView({block: "start", behavior: "smooth"});
                return;
            }
        }
    }

    /* Switch documents from outside the callback graph.
     *
     * set_props is resolved at CLICK time, never at install time: this file is evaluated before
     * dash-renderer has defined it. Falls back to clicking the hidden proxy button, which routes the
     * change through an ordinary callback. */
    function navigate(slug) {
        const dc = window.dash_clientside;
        if (dc && typeof dc.set_props === "function") {
            dc.set_props("docs-active", {data: slug});
            return;
        }
        window.__docsNavTarget = slug;
        const proxy = document.getElementById("docs-nav-proxy");
        if (proxy) proxy.click();
    }

    /* Links inside a rendered document. Three outcomes: an in-page anchor scrolls, an absolute URL
     * opens in a new tab (never navigating the widget away), and a relative path either switches the
     * viewer -- when it names one of the shipped documents -- or opens that file on GitHub. */
    function onLinkClick(e) {
        const a = e.target.closest && e.target.closest("#docs-body a[href]");
        if (!a) return;
        // The ATTRIBUTE, not a.href: the property is resolved to an absolute URL by the browser,
        // which would make every relative link look absolute.
        const href = a.getAttribute("href");
        if (!href) return;
        e.preventDefault();
        e.stopPropagation();

        if (href.charAt(0) === "#") return scrollToHeading(href.slice(1));
        if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.slice(0, 2) === "//") {
            window.open(href, "_blank", "noopener");
            return;
        }
        const rel = href.replace(/^\.\//, "").replace(/^\//, "").split("#")[0];
        const p = payloadCache || {};
        const slug = (p.links || {})[rel];
        if (slug) return navigate(slug);
        window.open((p.repo_blob || "") + rel, "_blank", "noopener");
    }

    /* Esc closes, but only when the overlay is open -- otherwise Leaflet's own Esc handling breaks.
     * It clicks the real close button rather than writing the class directly, which keeps toggle()
     * the single writer of the overlay's state. */
    function onKey(e) {
        if (e.key !== "Escape") return;
        const ov = document.getElementById("docs-overlay");
        if (!ov || ov.className.indexOf("is-open") === -1) return;
        e.preventDefault();
        const btn = document.getElementById("docs-close-btn");
        if (btn) btn.click();
    }

    // Capture phase, so react-markdown's own handlers cannot get there first.
    document.addEventListener("click", onLinkClick, true);
    document.addEventListener("keydown", onKey);

    window.dash_clientside.docs = {
        toggle: function (_open, _close, _backdrop) {
            return trig() === "docs-open-btn" ? "docs-overlay is-open" : "docs-overlay";
        },

        activeDoc: function () {
            const current = arguments[arguments.length - 1];  // the State
            const t = trig();
            if (t === "docs-open-btn") return current || "about";
            if (typeof t === "string" && t.indexOf(NAV_PREFIX) === 0) return t.slice(NAV_PREFIX.length);
            return NO();
        },

        render: function (slug, payload) {
            if (!payload) return NO();
            payloadCache = payload;
            const tabs = payload.order.map((s) => "docs-nav-item" + (s === slug ? " is-active" : ""));
            if (!slug) return [NO()].concat(tabs);

            const body = [];
            // A doc that declares an embed gets the player above its text, so its markdown reads as
            // a caption. Built as a component because dcc.Markdown strips raw HTML -- an <iframe>
            // written into the .md would disappear without an error.
            const embed = (payload.embed || {})[slug];
            if (embed) {
                body.push(B().h("Div", {
                    className: "docs-video",
                    children: B().h("Iframe", {
                        src: embed,
                        title: "Project presentation",
                        allow: "accelerometer; clipboard-write; encrypted-media; picture-in-picture; fullscreen",
                        referrerPolicy: "strict-origin-when-cross-origin",
                    }),
                }));
            }
            body.push(B().dcc("Markdown", {
                children: payload.text[slug],
                mathjax: !!payload.mathjax[slug],
                link_target: "_self",  // intercepted below; _self keeps the raw href inspectable
            }));
            const md = body.length === 1 ? body[0] : body;
            // A switch must start at the top rather than inheriting the previous document's scroll.
            // The one impure line in this file; it has to run after React commits.
            setTimeout(() => {
                const body = document.getElementById("docs-body");
                if (body) body.scrollTop = 0;
            }, 0);
            return [md].concat(tabs);
        },
    };
})();
