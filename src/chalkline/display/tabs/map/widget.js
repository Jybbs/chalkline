/**
 * Career pathway map rendered with D3.
 *
 * Columnar node-link diagram where columns represent Job Zone levels,
 * rows group by sector, and edges show career transitions weighted by
 * cosine similarity. Click events sync back to Python via traitlets.
 *
 * Static styling (fonts, sizes, foreground/muted/highlight colors)
 * lives in `app/chalkline.css` under the `.cl-pathway-map` block.
 * This file only handles structure, geometry, data-driven attributes,
 * and reactive state.
 */

import * as d3 from "https://esm.sh/d3@7";

const curvedLine = d3.line().curve(d3.curveBasis);


/* ------------------------------------------------------------------ */
/*  Renderer                                                          */
/* ------------------------------------------------------------------ */

export default {
    render({ model, el }) {

        function draw() {
            el.innerHTML = "";
            const data = JSON.parse(model.get("graph_data"));
            if (!data.nodes) return;

            const { columns, dimensions, edges, matched_x, matched_y,
                    nodes, total_height, total_width, you_are_here_text } = data;
            const { node_h, node_w, pad } = dimensions;
            const matchedId   = model.get("matched_id");
            const selectedId  = model.get("selected_id");
            const edgeOpacity = selectedId >= 0 ? 0.15 : 0.6;

            /* SVG container */

            const svg = d3.select(el).append("svg")
                .attr("class",       "cl-pathway-map")
                .attr("viewBox",     `${-pad} -10 ${total_width + 2 * pad} ${total_height + pad}`)
                .attr("width",       "100%")
                .style("max-height", `${Math.min(total_height + pad, 700)}px`);

            /* Arrow marker for advancement edges */

            svg.append("defs").append("marker")
                .attr("id",           "arrow")
                .attr("viewBox",      "0 0 10 10")
                .attr("refX",         10)
                .attr("refY",         5)
                .attr("markerWidth",  6)
                .attr("markerHeight", 6)
                .attr("orient",       "auto")
                .append("path")
                    .attr("class", "arrow-marker")
                    .attr("d",     "M 0 0 L 10 5 L 0 10 z");

            /* Column headers */

            svg.selectAll(".col-label").data(columns).join("text")
                .attr("class",       "col-label")
                .attr("x",           (d) => d.x)
                .attr("y",           12)
                .attr("text-anchor", "middle")
                .text((d) => d.label);

            /* Edges (only matched cluster's reach, pre-filtered in Python) */

            svg.selectAll(".edge").data(edges).join("path")
                .attr("class",            "edge")
                .attr("d",                (d) => curvedLine([[d.sx, d.sy], [d.mx, d.my], [d.tx, d.ty]]))
                .attr("stroke",           (d) => d.color)
                .attr("stroke-width",     (d) => Math.max(1.2, d.weight * 3))
                .attr("stroke-dasharray", (d) => d.is_cross_sector ? "6,4" : null)
                .attr("opacity",          edgeOpacity)
                .attr("marker-end",       (d) => d.is_advancement ? "url(#arrow)" : null);

            /* Credential badges (count pill on edges with credentials) */

            svg.selectAll(".cred-badge")
                .data(edges.filter((e) => e.credential_count > 0))
                .join("g")
                    .attr("class",     "cred-badge")
                    .attr("transform", (d) => `translate(${d.mx - 8}, ${d.my + 8})`)
                    .call((g) => {
                        g.append("circle")
                            .attr("class", "cred-badge-circle")
                            .attr("cx",    8)
                            .attr("cy",    8)
                            .attr("r",     9);
                        g.append("text")
                            .attr("class",       "cred-badge-text")
                            .attr("x",           8)
                            .attr("y",           12)
                            .attr("text-anchor", "middle")
                            .text((d) => d.credential_count);
                    });

            /* Matched node glow (inserted before nodes so it renders behind) */

            svg.append("rect")
                .attr("class",  "matched-glow")
                .attr("x",      matched_x - node_w / 2 - 4)
                .attr("y",      matched_y - node_h / 2 - 4)
                .attr("width",  node_w + 8)
                .attr("height", node_h + 8)
                .attr("rx",     10);

            /* Node card groups */

            const nodeG = svg.selectAll(".node").data(nodes).join("g")
                .attr("class",       "node")
                .classed("selected", (d) => d.id === selectedId)
                .classed("matched",  (d) => d.id === matchedId)
                .attr("transform",   (d) => `translate(${d.x - node_w / 2}, ${d.y - node_h / 2})`)
                .attr("opacity",     (d) => d.opacity)
                .on("click", (_event, d) => {
                    model.set("selected_id", d.id);
                    model.save_changes();
                });

            nodeG.append("rect")
                .attr("class",  "node-rect")
                .attr("width",  node_w)
                .attr("height", node_h)
                .attr("rx",     6);

            nodeG.append("rect")
                .attr("width",  4)
                .attr("height", node_h)
                .attr("rx",     2)
                .attr("fill",   (d) => d.color);

            nodeG.append("text")
                .attr("class", "node-title")
                .attr("x",     12)
                .attr("y",     22)
                .text((d) => d.title);

            nodeG.append("text")
                .attr("class", "node-subtitle")
                .attr("x",     12)
                .attr("y",     40)
                .text((d) => d.subtitle);

            /* "You are here" label above matched node */

            svg.append("text")
                .attr("class",       "you-are-here")
                .attr("x",           matched_x)
                .attr("y",           matched_y - node_h / 2 - 8)
                .attr("text-anchor", "middle")
                .text(you_are_here_text);
        }

        draw();
        for (const trait of ["graph_data", "selected_id"]) model.on(`change:${trait}`, draw);
    },
};
