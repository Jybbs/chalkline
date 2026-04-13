/**
 * Force-directed career pathway map rendered with D3.
 *
 * Nodes positioned by salary-driven force simulation. Tier-1 cards
 * (hop 0-1) show a match donut + title + subtitle. Tier-2 circles
 * (hop 2+) use a progress-ring border. The matched node renders as
 * an enriched hero card within the SVG.
 *
 * Static styling lives in `app/chalkline.css` under `.cl-pathway-map`.
 */

import * as d3 from "https://esm.sh/d3@7";


/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function addDonut(sel, cx, cy, labelCls, r, thick, yOff) {
    sel.append("circle").attr("class", "donut-bg")
        .attr("cx", cx).attr("cy", cy).attr("r", r);
    sel.append("path").attr("class", "donut-arc")
        .attr("transform", `translate(${cx}, ${cy})`)
        .attr("d", (d) => ringPath(d.match_pct, r, thick));
    sel.append("text").attr("class", labelCls)
        .attr("x", cx).attr("y", cy + yOff).attr("text-anchor", "middle")
        .text((d) => `${d.match_pct}%`);
}


function ringPath(pct, r, thick) {
    return d3.arc().innerRadius(r - thick).outerRadius(r).startAngle(0)
        .endAngle(Math.max(0.01, pct / 100) * 2 * Math.PI)();
}


function wageLabel(w) {
    return !w ? "" : w >= 1000 ? `$${Math.round(w / 1000)}k` : `$${w}`;
}


function wrapText(sel, maxW) {
    sel.each(function () {
        const el    = d3.select(this);
        const words = el.text().split(/\s+/);
        const x     = +el.attr("x");
        let line    = [];
        let tspan   = el.text(null).append("tspan").attr("x", x).attr("dy", 0);
        for (const word of words) {
            line.push(word);
            tspan.text(line.join(" "));
            if (tspan.node().getComputedTextLength() > maxW && line.length > 1) {
                line.pop();
                tspan.text(line.join(" "));
                line = [word];
                tspan = el.append("tspan").attr("x", x).attr("dy", "1.15em").text(word);
            }
        }
    });
}


/* ------------------------------------------------------------------ */
/*  Renderer                                                          */
/* ------------------------------------------------------------------ */

export default {
    render({ model, el }) {

        function draw() {
            el.innerHTML = "";
            if (!el.clientWidth) return;
            const raw = JSON.parse(model.get("graph_data"));
            if (!raw.nodes) return;

            const {
                dimensions: dims,
                edges,
                hero,
                nodes,
                wage_range
            } = raw;
            const mid = model.get("matched_id");
            const sid = model.get("selected_id");

            const clamp      = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
            const eid        = (v) => typeof v === "object" ? v.id : v;
            const select     = (_, d) => {
                model.set("selected_id", d.id);
                model.save_changes();
            };
            const tspanCount = (sel) => sel.selectAll("tspan").size() || 1;

            const {
                card_h   : cH,
                card_w   : cW,
                circle_r : cR,
                height   : H,
                hero_h   : hH,
                hero_w   : hW,
                pad
            } = dims;
            const W      = el.clientWidth || dims.width;
            const donutR = 17;

            /* ── Scales ─────────────────────────────────────────── */

            const we = [
                Math.floor(wage_range[0] / 5000) * 5000,
                Math.ceil(wage_range[1] / 5000) * 5000
            ];
            const xScale = d3.scaleLinear().domain(we)
                .range([pad + hW / 2 + 20, W - pad - cW / 2 - 10]);

            const sectors = [...new Set(nodes.map((n) => n.sector))].sort();
            const bandH   = (H - 2 * pad - 40) / Math.max(sectors.length, 1);
            const sY      = Object.fromEntries(
                sectors.map((s, i) => [s, pad + bandH * (i + 0.5)])
            );

            /* ── Force simulation ───────────────────────────────── */

            const noWageX = pad + 40;
            const sn      = nodes.map((n) => ({
                ...n,
                x : n.wage ? xScale(n.wage) : noWageX,
                y : sY[n.sector] || H / 2
            }));

            const byId = Object.fromEntries(sn.map((n) => [n.id, n]));

            const sl = edges.filter((e) => byId[e.source] && byId[e.target]);

            d3.forceSimulation(sn)
                .force("charge",  d3.forceManyBody().strength(-150))
                .force("collide", d3.forceCollide((d) =>
                    d.id === mid ? Math.max(hW, hH) / 2 + 28
                    : d.tier === 1 ? Math.max(cW, cH) / 2 + 26
                    : cR + 22
                ).strength(0.9).iterations(10))
                .force("link",    d3.forceLink(sl).id((d) => d.id)
                    .strength(0.03).distance(180))
                .force("x",       d3.forceX((d) =>
                    d.wage ? xScale(d.wage) : noWageX).strength(0.6))
                .force("y",       d3.forceY((d) => sY[d.sector] || H / 2).strength(0.1))
                .stop()
                .tick(500);

            sn.forEach((n) => {
                const [ex, ey] = n.id === mid ? [hW / 2, hH / 2]
                    : n.tier === 1 ? [cW / 2, cH / 2]
                    : [cR, cR];
                n.x = clamp(n.x, pad + ex, W - pad - ex);
                n.y = clamp(n.y, pad + ey, H - pad - ey - 40);
            });

            /* ── SVG ────────────────────────────────────────────── */

            const svg = d3.select(el).append("svg")
                .attr("class", "cl-pathway-map")
                .attr("viewBox", `0 0 ${W} ${H}`)
                .attr("width", "100%")
                .style("max-height", `${H}px`);

            /* ── Salary axis ────────────────────────────────────── */

            const aY = H - 20;

            svg.append("line").attr("class", "salary-shaft")
                .attr("x1", xScale(we[0]) - 20).attr("x2", xScale(we[1]) + 20)
                .attr("y1", aY).attr("y2", aY);

            d3.range(we[0], we[1] + 1, 10000).forEach((t) => {
                const x = xScale(t);
                svg.append("line").attr("class", "salary-tick-mark")
                    .attr("x1", x).attr("x2", x).attr("y1", aY - 4).attr("y2", aY + 4);
                svg.append("text").attr("class", "salary-label")
                    .attr("x", x).attr("y", aY + 16).attr("text-anchor", "middle")
                    .text(wageLabel(t));
            });

            /* ── Edges (1-hop matched only, uniform style) ──────── */

            const edgeSel = svg.selectAll(".edge")
                .data(sl.filter((e) => eid(e.source) === mid || eid(e.target) === mid))
                .join("path")
                .attr("class", "edge")
                .attr("d", (d) => {
                    const s  = byId[eid(d.source)];
                    const t  = byId[eid(d.target)];
                    const mx = (s.x + t.x) / 2;
                    const my = (s.y + t.y) / 2 - 20;
                    return `M${s.x},${s.y} Q${mx},${my} ${t.x},${t.y}`;
                })
                .attr("stroke", (d) => d.color)
                .attr("stroke-width", (d) => Math.max(1.5, d.weight * 3))
                .attr("opacity", 0.5);

            /* ── Tier 2 circles ─────────────────────────────────── */

            const t2G = svg.selectAll(".node-circle")
                .data(sn.filter((n) => n.tier === 2)).join("g")
                .attr("class", "node-circle")
                .classed("selected", (d) => d.id === sid)
                .attr("transform", (d) => `translate(${d.x}, ${d.y})`)
                .attr("opacity", 0.4)
                .on("click", select);

            t2G.append("circle").attr("class", "circle-bg").attr("r", cR);
            t2G.append("path").attr("class", "progress-ring")
                .attr("d", (d) => ringPath(d.match_pct, cR, 2.5));
            t2G.append("text").attr("class", "circle-pct")
                .attr("y", 4).attr("text-anchor", "middle")
                .text((d) => `${d.match_pct}%`);
            t2G.append("text").attr("class", "circle-label")
                .attr("x", 0)
                .attr("y", (d) => d.y < H / 2 ? cR + 14 : -(cR + 6))
                .attr("text-anchor", "middle")
                .text((d) => d.title)
                .call(wrapText, cR * 3.5);

            /* Shift above-labels up so wrapped lines stack away from circle */
            t2G.each(function (d) {
                if (d.y >= H / 2) {
                    const label = d3.select(this).select(".circle-label");
                    label.attr("y", +label.attr("y") - (tspanCount(label) - 1) * 8);
                }
            });

            /* Dim null-wage tier-2 circles */
            t2G.filter((d) => !d.wage).attr("opacity", 0.25)
                .select(".circle-bg").attr("stroke-dasharray", "3 2");

            /* ── Tier 1 cards ───────────────────────────────────── */

            const t1G = svg.selectAll(".node")
                .data(sn.filter((n) => n.tier === 1 && n.id !== mid)).join("g")
                .attr("class", "node")
                .classed("selected", (d) => d.id === sid)
                .attr("transform", (d) =>
                    `translate(${d.x - cW / 2}, ${d.y - cH / 2})`)
                .on("click", select);

            t1G.append("rect").attr("class", "node-rect")
                .attr("width", cW).attr("height", cH).attr("rx", 6);
            t1G.append("rect").attr("width", 4).attr("height", cH).attr("rx", 2)
                .attr("fill", (d) => d.color);

            /* Dim null-wage tier-1 cards */
            t1G.filter((d) => !d.wage).attr("opacity", 0.5)
                .select(".node-rect").attr("stroke-dasharray", "4 2")
                .attr("stroke", "var(--cl-muted-foreground)").attr("stroke-width", 1);

            const dnX = donutR + 10;
            const dnY = cH / 2;
            addDonut(t1G, dnX, dnY, "donut-label", donutR, 3, 4);

            /* Title + subtitle (vertically centered as a block) */
            const ttX = dnX + donutR + 8;
            t1G.append("text").attr("class", "node-title")
                .attr("x", ttX).attr("y", 0)
                .text((d) => d.title).call(wrapText, cW - ttX - 6);
            t1G.append("text").attr("class", "node-subtitle")
                .attr("x", ttX).attr("y", 0)
                .text((d) => d.subtitle);

            t1G.each(function () {
                const g     = d3.select(this);
                const lines = tspanCount(g.select(".node-title"));
                const y0    = (cH - (lines * 13 + 14)) / 2 + 11;
                g.select(".node-title").attr("y", y0);
                g.select(".node-subtitle").attr("y", y0 + lines * 13 + 2);
            });

            /* ── Hero card ──────────────────────────────────────── */

            const hn = byId[mid];
            if (hn) {
                const hx = hn.x - hW / 2;
                const hy = hn.y - hH / 2;

                svg.append("rect").attr("class", "matched-glow")
                    .attr("x", hx - 5).attr("y", hy - 5)
                    .attr("width", hW + 10).attr("height", hH + 10).attr("rx", 10);

                const hg = svg.append("g").datum(hn).attr("class", "hero-card")
                    .attr("transform", `translate(${hx}, ${hy})`)
                    .style("cursor", "pointer")
                    .on("click", select);

                hg.append("rect").attr("class", "hero-bg")
                    .attr("width", hW).attr("height", hH).attr("rx", 8);
                hg.append("rect").attr("width", 5).attr("height", hH).attr("rx", 2)
                    .attr("fill", hero.sector_color);

                /* Match count (top right) */
                if (hero.n_matches > 0) {
                    hg.append("text").attr("class", "hero-nav")
                        .attr("x", hW - 12).attr("y", 18).attr("text-anchor", "end")
                        .text(`${hero.n_matches} \u2192`);
                }

                const hdR = 22;
                const hdX = hdR + 14;
                const hdY = hH / 2;
                addDonut(hg, hdX, hdY, "hero-donut-label", hdR, 3.5, 5);

                /* Title + subtitle (vertically centered) */
                const htX = hdX + hdR + 10;
                const heroSub = `${hero.size} postings`
                    + (hero.wage ? ` \u00b7 ${wageLabel(hero.wage)}` : "");

                hg.append("text").attr("class", "hero-title")
                    .attr("x", htX).attr("y", 0)
                    .text(hero.title).call(wrapText, hW - htX - 50);

                const heroLines = tspanCount(hg.select(".hero-title"));
                const hy0       = (hH - (heroLines * 16 + 14)) / 2 + 13;
                hg.select(".hero-title").attr("y", hy0);

                hg.append("text").attr("class", "node-subtitle")
                    .attr("x", htX).attr("y", hy0 + heroLines * 16 + 2)
                    .text(heroSub);
            }

            /* ── Tooltip ────────────────────────────────────────── */

            const tip = svg.append("g").attr("class", "cl-tooltip")
                .attr("visibility", "hidden");

            function showTip(_, d) {
                if (d.id === mid) return;
                tip.selectAll("*").remove();

                const p  = 12;
                const lh = 15;
                const tw = 230;
                let y = p;

                /* Full untruncated title */
                tip.append("text").attr("class", "tooltip-title")
                    .attr("x", p).attr("y", y += lh)
                    .text(d.full_title).call(wrapText, tw - p * 2);

                const titleLines = tspanCount(tip.select(".tooltip-title"));
                if (titleLines > 1) y += (titleLines - 1) * lh;

                tip.append("text").attr("class", "tooltip-dim")
                    .attr("x", p).attr("y", y += lh)
                    .text(d.hop != null
                        ? `${d.hop} step${d.hop !== 1 ? "s" : ""} away`
                        : "Not connected");

                /* Wage bars */
                const hw = hero.wage;
                const nw = d.wage;
                if (hw && nw) {
                    y += 6;
                    const mx = Math.max(hw, nw);
                    const bm = tw - p * 2 - 60;

                    for (const [label, cls, w] of [
                        ["You",  "tooltip-bar-you",  hw],
                        ["This", "tooltip-bar-dest", nw]
                    ]) {
                        const bw = (w / mx) * bm;
                        tip.append("text").attr("class", "tooltip-dim")
                            .attr("x", p).attr("y", y += lh).text(label);
                        tip.append("rect").attr("class", cls)
                            .attr("x", p + 40).attr("y", y - 9)
                            .attr("width", bw).attr("height", 7).attr("rx", 3);
                        tip.append("text").attr("class", "tooltip-dim")
                            .attr("x", p + 44 + bw).attr("y", y).text(wageLabel(w));
                    }

                    const delta = nw - hw;
                    if (delta !== 0) {
                        tip.append("text")
                            .attr("class", delta > 0
                                ? "tooltip-positive"
                                : "tooltip-negative")
                            .attr("x", p).attr("y", y += lh)
                            .text((delta > 0 ? "+" : "")
                                + wageLabel(Math.abs(delta)) + "/yr");
                    }
                }

                y += p;
                tip.insert("rect", ":first-child").attr("class", "tooltip-bg")
                    .attr("width", tw).attr("height", y).attr("rx", 8);

                const nudge = (d.tier === 1 ? cW / 2 : cR) + 10;
                let tx = d.x + nudge;
                if (tx + tw > W - pad) tx = d.x - tw - nudge;
                const ty = clamp(d.y - y / 2, pad, H - pad - y);

                tip.attr("transform", `translate(${tx}, ${ty})`)
                    .attr("visibility", "visible");
                edgeSel.attr("opacity", (e) =>
                    eid(e.source) === d.id || eid(e.target) === d.id ? 0.85 : 0.1
                );
            }

            function hideTip() {
                tip.attr("visibility", "hidden");
                edgeSel.attr("opacity", 0.5);
            }

            svg.selectAll(".node, .node-circle")
                .on("mouseenter", showTip).on("mouseleave", hideTip);
        }

        const ro = new ResizeObserver(draw);
        ro.observe(el);
        for (const t of ["graph_data", "selected_id"]) model.on(`change:${t}`, draw);
    },
};
