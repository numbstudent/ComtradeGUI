import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

import_or_install('flask')


from flask import Flask
import os

from flask import Flask, flash, request, redirect, url_for, render_template

import comtradeanalysis as ca

UPLOAD_FOLDER = './upload/'
# UPLOAD_FOLDER = join(dirname(realpath(__file__)), '../comtradefiles/')

ALLOWED_EXTENSIONS = {'dat', 'cfg'}

app = Flask(__name__,template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_mapping(
    SECRET_KEY='awikwokprod',
    # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)

@app.route('/')
def hello_world():
    return 'Hello World'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'datfile' not in request.files or 'cfgfile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        datfile = request.files['datfile']
        cfgfile = request.files['cfgfile']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if datfile.filename == '':
            flash('No selected dat file')
            return redirect(request.url)
        if cfgfile.filename == '':
            flash('No selected cfg file')
            return redirect(request.url)
        if cfgfile.filename.rsplit('.', 1)[1].lower() != 'cfg':
            flash('Incorrect cfg file')
            return redirect(request.url)
        if datfile.filename.rsplit('.', 1)[1].lower() != 'dat':
            flash('Incorrect dat file')
            return redirect(request.url)
        if datfile and allowed_file(datfile.filename) and cfgfile and allowed_file(cfgfile.filename):
            datfilename = 'toanalyze.dat'
            cfgfilename = 'toanalyze.cfg'
            datpath = os.path.join(UPLOAD_FOLDER,datfilename)
            datfile.save(datpath)
            cfgpath = os.path.join(UPLOAD_FOLDER,cfgfilename)
            cfgfile.save(cfgpath)
            return redirect(url_for('upload_file'))
    return render_template('/index.html')

if __name__ == '__main__':
    # app.run()
    app.run(debug = True)
