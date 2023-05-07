const ContinuousVisualization = function(width, height, context) {
	this.draw = function(objects) {
		for (const p of objects) {
			if (p.Shape == "rect")
				this.drawRectange(p.x, p.y, p.w, p.h, p.Color, p.Filled);
			else if (p.Shape == "circle")
				this.drawCircle(p.x, p.y, p.r, p.Color, p.Filled);
			else 
				this.drawCustomImage(p.Shape, p.x, p.y, p.w, p.h);
		};
	};

	this.drawCircle = function(x, y, radius, color, fill) {
		const cx = x * width;
		const cy = y * height;
		const r = radius;

		context.beginPath();
		context.arc(cx, cy, r, 0, Math.PI * 2, false);
		context.closePath();

		context.strokeStyle = color;
		context.stroke();

		if (fill) {
			context.fillStyle = color;
			context.fill();
		}

	};

	this.drawRectange = function(x, y, w, h, color, fill) {
		context.beginPath();
		const dx = w * width;
		const dy = h * height;

		// Keep the drawing centered:
		const x0 = (x*width) - 0.5*dx;
		const y0 = (y*height) - 0.5*dy;

		context.strokeStyle = color;
		context.fillStyle = color;
		if (fill)
			context.fillRect(x0, y0, dx, dy);
		else
			context.strokeRect(x0, y0, dx, dy);
	};

	this.drawCustomImage = function (shape, x, y, w, h, text, text_color_) {
		const img = new Image();
		img.src = "local/custom/".concat(shape);

		// Calculate coordinates so the image is always centered
		const cx = x * width - w / 2;
		const cy = y * height - h / 2;

		img.onload = function () {
			context.drawImage(img, cx, cy, w, h);
			// This part draws the text on the image
			if (text !== undefined) {
				// ToDo: Fix fillStyle
				// context.fillStyle = text_color;
				context.textAlign = "center";
				context.textBaseline = "middle";
				context.fillText(text, tx, ty);
			}
		};
	};

	this.resetCanvas = function() {
		context.clearRect(0, 0, width, height);
		context.beginPath();
	};
};

const Simple_Continuous_Module = function(canvas_width, canvas_height) {
	// Create the element
	// ------------------

	// background - image: url("https://m.media-amazon.com/images/I/91dEOYkZhzL._AC_UF894,1000_QL80_.jpg"); \

  const canvas = document.createElement("canvas");
  Object.assign(canvas, {
    width: canvas_width,
    height: canvas_height,
	  style: 'border:1px dotted; \
		background-image: url("local/custom/boid_flockers/resources/roadmap.png"); \
		background-repeat: no-repeat; \
		background-size: cover;'
  });
	// Append it to body:
  document.getElementById("elements").appendChild(canvas);

	// Create the context and the drawing controller:
	const context = canvas.getContext("2d");
	const canvasDraw = new ContinuousVisualization(canvas_width, canvas_height, context);

	this.render = function(data) {
		canvasDraw.resetCanvas();
		canvasDraw.draw(data);
	};

	this.reset = function() {
		canvasDraw.resetCanvas();
	};
};
