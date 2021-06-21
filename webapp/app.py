from flask import Flask, render_template, request
from matplotlib import pyplot

from evaluate import Evaluator

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

pix2pix_evaluator = Evaluator("pix2pix_gan_model.h5")
cycle_gan_evaluator = Evaluator("cycle_gan_model.h5", True)

input_path = 'static/input.png'
pix2pix_output_path = 'static/pix2pix_output.png'
cycle_gan_output_path = 'static/cycle_gan_output.png'


@app.route('/')
@app.route('/gan')
def gan():
    return render_template("gan.html")


@app.route('/gan/upload', methods=['POST'])
def gan_upload():
    is_image = request.args.get('type') == 'image'
    file = request.files['file']
    file.save(input_path)

    if is_image:
        pix2pix_generated_image_arr = pix2pix_evaluator.predict(input_path, (512, 512), plot=False)
        pyplot.imsave(pix2pix_output_path, pix2pix_generated_image_arr)
        cycle_gan_generated_image_arr = cycle_gan_evaluator.predict(input_path, (512, 512), plot=False)
        pyplot.imsave(cycle_gan_output_path, cycle_gan_generated_image_arr)
    else:
        raise Exception("Unsupported file type")

    return render_template("gan.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001, debug=True)
