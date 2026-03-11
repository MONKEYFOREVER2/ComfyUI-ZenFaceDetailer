import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../scripts/ui.js";

app.registerExtension({
    name: "Zen.FaceDetailer",
    async setup() {
        // Load custom CSS
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = api.apiURL("/extensions/ComfyUI-ZenFaceDetailer/css/zen_face_detailer.css");
        document.head.appendChild(link);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ZenFaceDetailer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Color the node itself for a "Zen" aesthetic
                this.color = "#15161A";
                this.bgcolor = "#1E1F26";
                this.title_color = "#E0E5FF";
                this.shape = LiteGraph.ROUND_SHAPE;

                // Set initial size
                this.size = [360, 600];

                // We're going to create a modern DOM widget to act as a settings panel
                // This allows HTML/CSS styling which looks infinitely better than canvas drawing.
                const zenContainer = $el("div.zen-container", [
                    $el("div.zen-header", [
                        $el("h3", { textContent: "ZenFaceDetailer" }),
                        $el("span.zen-badge", { textContent: "v1.1" })
                    ]),
                    $el("div.zen-tabs", [
                        $el("button.zen-tab-btn.active", { textContent: "Main", onclick: (e) => switchTab(e, 'main') }),
                        $el("button.zen-tab-btn", { textContent: "Face", onclick: (e) => switchTab(e, 'face') }),
                        $el("button.zen-tab-btn", { textContent: "Sampling", onclick: (e) => switchTab(e, 'sampling') })
                    ])
                ]);

                // Add the wrapper to hide Comfy's default widgets, we will bind to them later
                const widgetWrapper = this.addDOMWidget("zen_ui", "zen_ui", zenContainer, {
                    serialize: false,
                    hideOnZoom: false
                });

                // Set node properties
                this.properties = this.properties || {};

                function switchTab(e, tabName) {
                    // Update active tab styling
                    const btns = zenContainer.querySelectorAll('.zen-tab-btn');
                    btns.forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');

                    // Dispatch custom event to tell CSS which tab is active (if we want to use CSS selectors)
                    zenContainer.setAttribute("data-active-tab", tabName);
                }

                zenContainer.setAttribute("data-active-tab", "main");

                setTimeout(() => {
                    // Find all internal comfy widgets and hide them from the canvas
                    // We will just let them draw but shrink them, or we can build our own HTML inputs that mirror them
                    this.widgets.forEach(w => {
                        if (w.name !== "zen_ui") {
                            // Optionally hide them visually so we can replace them with HTML.
                            // However, directly mirroring 20 widgets into HTML is complex.
                            // A better "beautiful node" approach without breaking LiteGraph serialize is to just skin the node 
                            // and let LiteGraph draw the widgets but with better paddings, using HTML just for the header/presets!
                        }
                    });
                }, 100);

                return r;
            };

            // Override drawing to make it look smooth and modern
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function (ctx) {
                if (onDrawBackground) onDrawBackground.apply(this, arguments);

                if (this.flags.collapsed) return;

                const w = this.size[0];
                const h = this.size[1];

                // Draw a beautiful soft gradient over the node background
                const grad = ctx.createLinearGradient(0, 0, w, h);
                grad.addColorStop(0, "#252B3B");
                grad.addColorStop(1, "#181A22");

                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.roundRect(0, 0, w, h, 8);
                ctx.fill();

                // Draw subtle border
                ctx.strokeStyle = "#4D5B80";
                ctx.lineWidth = 1;
                ctx.stroke();
            };
        }
    }
});
