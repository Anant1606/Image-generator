from flask import Flask, request, render_template, send_file
import io
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = generate_image(prompt, image_gen_model)
        
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
