import os
import numpy as np
import seaborn as sns
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt

from wavefront import load_obj
from core import Core


UPLOAD_FOLDER = './test_shapes'
ALLOWED_EXTENSIONS = set(['obj'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def draw_activations(act, num2label):
    fig = plt.figure(figsize=(5, 5))
    sns_plot = sns.barplot(x=num2label, y=act)
    full_filename = './static/activations.png'
    fig = sns_plot.get_figure()
    fig.savefig(full_filename)


def draw_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    full_filename = './static/points.png'
    fig.savefig(full_filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            pointcloud = load_obj(file_path).vertices
            pointcloud = np.asarray(pointcloud)
            res = core.predict(pointcloud)
            res = res.detach().cpu().numpy()[0]

            draw_activations(res, list(core.label2num.keys()))
            draw_points(pointcloud)

            return render_template('index2.html',
                                   activations_image='activations.png',
                                   points_image='points.png')

    return render_template('index.html')


if __name__ == '__main__':
   core = Core(device='cpu')
   app.run()

