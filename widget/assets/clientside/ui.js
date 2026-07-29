/* Pure-UI clientside callbacks: no bundle data, just prop and style writes.
 *
 * These replace Python callbacks that only ever returned a constant or a formatted string. In a static build a server callback POSTs to an endpoint that is not there and fails silently, so the control renders and does nothing -- see widget/static/export.py.
 *
 * Every function is registered from Python via ClientsideFunction(namespace="ui", ...), so the dependency graph stays greppable in the component modules rather than living only in JS.
 *
 * Style dicts and URL tables are NOT redefined here. They arrive as State from dcc.Stores the layout fills from map_common, so Python remains the single definition and the two cannot drift.
 */

(function () {
    window.dash_clientside = window.dash_clientside || {};
    const NO = () => window.dash_clientside.no_update;
    const B = () => window.dash_clientside.bundle;
    const trig = () => window.dash_clientside.bundle.triggeredId();

    /* dash-leaflet reports the map centre as [lat, lon] on first render but {lat, lng} after any move, so both shapes have to be handled -- the Python original had the same quirk. */
    function latlon(center) {
        if (!center) return null;
        if (Array.isArray(center)) return [center[0], center[1]];
        return [center.lat, center.lng];
    }

    window.dash_clientside.ui = {
        tileUrl: function (selected, urls) {
            if (!selected || !urls) return NO();
            return urls[selected] || NO();
        },

        mapViewReadout: function (zoom, center) {
            const c = latlon(center);
            if (zoom === null || zoom === undefined || !c) return "zoom —  ·  center —";
            return `zoom ${zoom}  ·  center ${c[0].toFixed(4)}, ${c[1].toFixed(4)}`;
        },

        /* Debug: fly to a directly-typed zoom/centre. Blank fields keep the current value and an unparseable centre is ignored rather than moving the map somewhere arbitrary. */
        mapViewport: function (_n, zoomIn, centerStr, curCenter, curZoom) {
            let center = latlon(curCenter);
            if (centerStr) {
                const parts = String(centerStr).replace(/\s/g, "").split(",").map(Number);
                if (parts.length !== 2 || parts.some(Number.isNaN)) return NO();
                center = parts;
            }
            const zoom = zoomIn !== null && zoomIn !== undefined ? zoomIn : curZoom;
            if (center === null && (zoom === null || zoom === undefined)) return NO();
            return {center: center, zoom: zoom, transition: "flyTo"};
        },

        /* One function for all five help popups. Open and close are told apart by the id suffix -- every close button is "<x>-help-close-btn" against the opener's "<x>-help-btn" -- which avoids threading five per-popup constants through State just to compare one string. */
        helpPopup: function (_open, _close, consts) {
            const t = trig();
            return typeof t === "string" && t.endsWith("-close-btn") ? consts.help_hidden : consts.help_visible;
        },

        activeGraphSite: function (btnClicks, selectedUids, currentActive) {
            const sel = selectedUids || [];
            const t = trig();
            if (t && typeof t === "object" && t.type === "graph-site-btn") {
                if ((btnClicks || []).some((c) => c)) return t.index;
            }
            // selected-site changed: follow a newly added site, else fall back to the last one.
            if (sel.indexOf(currentActive) === -1) return sel.length ? sel[sel.length - 1] : null;
            if (sel.length && sel[sel.length - 1] !== currentActive) return sel[sel.length - 1];
            return NO();
        },

        graphOverlay: function (selectedUids, _close, graphToggle, consts) {
            if (trig() === "close-graph-btn") return consts.overlay_hidden;
            if ((graphToggle || []).indexOf("show") === -1) return consts.overlay_hidden;
            if (selectedUids && selectedUids.length) return consts.overlay_visible;
            return NO();
        },



        /* The selection set. Three ways in -- a map marker, a table row's ×, and "clear selection" -- so the trigger id decides which branch runs.
         *
         * The `.some(c => c)` guards matter: re-rendering the marker layer re-fires its ALL input with every n_clicks back at 0, which would otherwise read as a click on whichever marker Dash reports first. */
        selectedSites: function (markerClicks, removeClicks, _clear, current, activeMenu) {
            const cur = current || [];
            const t = trig();
            if (t === "clear-selection-btn") return [];

            if (t && typeof t === "object" && t.type === "iwqis-marker") {
                if (!(markerClicks || []).some((c) => c)) return NO();
                const uid = t.index;
                const held = cur.indexOf(uid) !== -1;
                // The Debug tab works one site at a time: clicking the selected site clears it, clicking another replaces it.
                if (activeMenu === "debug") return held ? [] : [uid];
                return held ? cur.filter((s) => s !== uid) : cur.concat([uid]);
            }

            if (t && typeof t === "object" && t.type === "remove-site-btn") {
                if (!(removeClicks || []).some((c) => c)) return NO();
                return cur.filter((s) => s !== t.index);
            }

            return NO();
        },

        /* (Pin drop moved to the forecast namespace, which snaps the marker to the reach outlet the
         * forecast is actually computed at -- see forecast.js regionGeom.) */

        switchMenu: function (_e, _f, _d, selectedSites, activeSite, consts) {
            const show = {display: "block"};
            const hide = {display: "none"};
            const on = consts.tab_active;
            const off = consts.tab_inactive;
            const t = trig();
            if (t === "menu-tab-debug") {
                // The Debug tab works one site at a time, so the selection is trimmed on entry.
                const trimmed = activeSite ? [activeSite] : (selectedSites || []).slice(0, 1);
                return [hide, hide, show, off, off, on, "debug", trimmed];
            }
            if (t === "menu-tab-forecast") {
                return [hide, show, hide, off, on, off, "forecast", NO()];
            }
            return [show, hide, hide, on, off, off, "explore", NO()];
        },


    };
})();
