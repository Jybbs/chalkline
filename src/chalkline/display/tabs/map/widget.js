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
/*  Rectangular bbox collision force (vendored from d3-bboxCollide)   */
/* ------------------------------------------------------------------ */

function bboxCollide(bbox) {
    var nodes, boxes, strength = 1, iterations = 1;

    function force() {
        for (var k = 0; k < iterations; ++k) {
            var corners = [];
            nodes.forEach(function (d, i) {
                var b = boxes[i];
                var cx = (b[0][0] + b[1][0]) / 2;
                var cy = (b[0][1] + b[1][1]) / 2;
                corners.push({node: d, vx: d.vx, vy: d.vy, x: d.x + cx, y: d.y + cy});
                corners.push({node: d, vx: d.vx, vy: d.vy, x: d.x + b[0][0], y: d.y + b[0][1]});
                corners.push({node: d, vx: d.vx, vy: d.vy, x: d.x + b[0][0], y: d.y + b[1][1]});
                corners.push({node: d, vx: d.vx, vy: d.vy, x: d.x + b[1][0], y: d.y + b[0][1]});
                corners.push({node: d, vx: d.vx, vy: d.vy, x: d.x + b[1][0], y: d.y + b[1][1]});
            });

            var tree = d3.quadtree(corners,
                (d) => d.x + d.vx, (d) => d.y + d.vy);

            for (var i = 0; i < corners.length; ++i) {
                var ni  = ~~(i / 5);
                var nd  = nodes[ni];
                var bi  = boxes[ni];
                var xi  = nd.x + nd.vx;
                var yi  = nd.y + nd.vy;
                var nx1 = xi + bi[0][0], ny1 = yi + bi[0][1];
                var nx2 = xi + bi[1][0], ny2 = yi + bi[1][1];
                var bW  = bi[1][0] - bi[0][0];
                var bH  = bi[1][1] - bi[0][1];

                tree.visit(function (quad, x0, y0, x1, y1) {
                    if (!quad.data) return x0 > nx2 || x1 < nx1 || y0 > ny2 || y1 < ny1;
                    if (quad.data.node.index === ni) return;

                    var other = quad.data.node;
                    var bj    = boxes[other.index];
                    var dx1   = other.x + other.vx + bj[0][0];
                    var dy1   = other.y + other.vy + bj[0][1];
                    var dx2   = other.x + other.vx + bj[1][0];
                    var dy2   = other.y + other.vy + bj[1][1];

                    if (nx1 > dx2 || dx1 > nx2 || ny1 > dy2 || dy1 > ny2) return;

                    var dW = bj[1][0] - bj[0][0];
                    var dH = bj[1][1] - bj[0][1];
                    var xO = bW + dW - (Math.max(nx2, dx2) - Math.min(nx1, dx1));
                    var yO = bH + dH - (Math.max(ny2, dy2) - Math.min(ny1, dy1));

                    if ((nx1 + nx2) / 2 < (dx1 + dx2) / 2) {
                        nd.vx    -= xO * strength * (yO / bH);
                        other.vx += xO * strength * (yO / dH);
                    } else {
                        nd.vx    += xO * strength * (yO / bH);
                        other.vx -= xO * strength * (yO / dH);
                    }
                    if ((ny1 + ny2) / 2 < (dy1 + dy2) / 2) {
                        nd.vy    -= yO * strength * (xO / bW);
                        other.vy += yO * strength * (xO / dW);
                    } else {
                        nd.vy    += yO * strength * (xO / bW);
                        other.vy -= yO * strength * (xO / dW);
                    }
                });
            }
        }
    }

    force.initialize = function (_) {
        nodes = _;
        boxes = new Array(nodes.length);
        for (var i = 0; i < nodes.length; ++i) boxes[i] = bbox(nodes[i], i, nodes);
    };

    force.iterations = function (_) {
        return arguments.length ? (iterations = +_, force) : iterations;
    };

    force.strength = function (_) {
        return arguments.length ? (strength = +_, force) : strength;
    };

    return force;
}


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

        const cache = new Map();

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
            const sn      = nodes.map((n) => {
                const prior = cache.get(`${mid}:${n.id}`);
                return {
                    ...n,
                    x : prior?.x ?? (n.wage ? xScale(n.wage) : noWageX),
                    y : prior?.y ?? (sY[n.sector] || H / 2)
                };
            });

            const byId     = Object.fromEntries(sn.map((n) => [n.id, n]));
            const sl       = edges.filter((e) => byId[e.source] && byId[e.target]);
            const anyFresh = sn.some((n) => !cache.has(`${mid}:${n.id}`));
            const ticks    = anyFresh ? 500 : 80;

            const gap = 12;
            d3.forceSimulation(sn)
                .force("charge",  d3.forceManyBody().strength(-150))
                .force("collide", bboxCollide((d) => {
                    if (d.id === mid)  return [[-(hW / 2 + gap), -(hH / 2 + gap)],
                                               [ (hW / 2 + gap),  (hH / 2 + gap)]];
                    if (d.tier === 1)  return [[-(cW / 2 + gap), -(cH / 2 + gap)],
                                               [ (cW / 2 + gap),  (cH / 2 + gap)]];
                    return [[-(cR + gap), -(cR + gap)], [cR + gap, cR + gap]];
                }).strength(0.9).iterations(10))
                .force("link",    d3.forceLink(sl).id((d) => d.id)
                    .strength(0.03).distance(180))
                .force("x",       d3.forceX((d) =>
                    d.wage ? xScale(d.wage) : noWageX).strength(0.6))
                .force("y",       d3.forceY((d) => sY[d.sector] || H / 2).strength(0.1))
                .stop()
                .tick(ticks);

            sn.forEach((n) => {
                const [ex, ey] = n.id === mid ? [hW / 2, hH / 2]
                    : n.tier === 1 ? [cW / 2, cH / 2]
                    : [cR, cR];
                n.x = clamp(n.x, pad + ex, W - pad - ex);
                n.y = clamp(n.y, pad + ey, H - pad - ey - 40);
                cache.set(`${mid}:${n.id}`, { x: n.x, y: n.y });
            });

            /* ── SVG ────────────────────────────────────────────── */

            const svg = d3.select(el).append("svg")
                .attr("class", "cl-pathway-map")
                .attr("viewBox", `0 0 ${W} ${H}`)
                .attr("width", "100%")
                .style("max-height", `${H}px`);

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

            /* Title + italic suffix + stats subtitle (vertically centered) */
            const ttX = dnX + donutR + 8;
            t1G.append("text").attr("class", "node-title")
                .attr("x", ttX).attr("y", 0)
                .text((d) => d.title).call(wrapText, cW - ttX - 6);
            t1G.filter((d) => d.suffix)
                .append("text").attr("class", "node-suffix")
                .attr("x", ttX).attr("y", 0)
                .text((d) => d.suffix);
            t1G.append("text").attr("class", "node-subtitle")
                .attr("x", ttX).attr("y", 0)
                .text((d) => d.subtitle);

            t1G.each(function (d) {
                const g       = d3.select(this);
                const lines   = tspanCount(g.select(".node-title"));
                const blockH  = lines * 13 + (d.suffix ? 13 : 0) + 13;
                const y0      = (cH - blockH) / 2 + 11;
                let cursor    = y0;
                g.select(".node-title").attr("y", cursor);
                cursor += lines * 13;
                if (d.suffix) {
                    g.select(".node-suffix").attr("y", cursor + 2);
                    cursor += 13;
                }
                g.select(".node-subtitle").attr("y", cursor + 2);
            });

            /* ── Hero card ──────────────────────────────────────── */

            const hn = byId[mid];
            if (hn) {
                const hx = hn.x - hW / 2;
                const hy = hn.y - hH / 2;

                svg.append("rect").attr("class", "matched-glow")
                    .attr("x", hx - 5).attr("y", hy - 5)
                    .attr("width", hW + 10).attr("height", hH + 10).attr("rx", 10)
                    .attr("stroke", hero.match_color);

                const hg = svg.append("g").datum(hn).attr("class", "hero-card")
                    .attr("transform", `translate(${hx}, ${hy})`)
                    .style("cursor", "pointer")
                    .on("click", select);

                hg.append("rect").attr("class", "hero-bg")
                    .attr("width", hW).attr("height", hH).attr("rx", 8)
                    .attr("stroke", hero.match_color).attr("stroke-width", 2.5);
                hg.append("rect").attr("width", 5).attr("height", hH).attr("rx", 2)
                    .attr("fill", hero.sector_color);


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
