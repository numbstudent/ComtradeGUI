import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

import_or_install('flask')


from flask import Flask
import os
import re

from flask import Flask, flash, request, redirect, url_for, render_template

import comtradeanalysis as ca

# UPLOAD_FOLDER = './upload/'
UPLOAD_FOLDER = './'
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
    return redirect(url_for('upload_file'))
    # return 'Hello World'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    data = {}
    channels = []
    error = None
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
        if datfile.filename.rsplit('.', 1)[0] != cfgfile.filename.rsplit('.', 1)[0]:
            flash('The dat filename is different to cfg filename')
            return redirect(request.url)
        if datfile and allowed_file(datfile.filename) and cfgfile and allowed_file(cfgfile.filename):
            datfilename = re.sub(r'[^\w]', ' ', cfgfile.filename)
            datfilename = datfilename[:-3]+".dat"
            cfgfilename = re.sub(r'[^\w]', ' ', datfile.filename)
            cfgfilename = cfgfilename[:-3]+".cfg"
            
            dir = UPLOAD_FOLDER
            for file in os.listdir(dir):
                if file.endswith(".cfg"):
                    os.remove(os.path.join(dir, file))
                if file.endswith(".dat"):
                    os.remove(os.path.join(dir, file))
            datpath = os.path.join(UPLOAD_FOLDER,datfilename)
            datfile.save(datpath)
            cfgpath = os.path.join(UPLOAD_FOLDER,cfgfilename)
            cfgfile.save(cfgpath)
            try:
                channels = ca.read_comtrade_channels()
            except:
                channels = []
                error = "The cfg / dat file cannot be read."
                flash(error)
            return redirect(url_for('upload_file'))
    else:
        channels = ca.read_comtrade_channels()
        
    data["cfgfilename"] = None
    data["datfilename"] = None
    dir = UPLOAD_FOLDER
    for file in os.listdir(dir):
        if file.endswith(".cfg"):
            data["cfgfilename"] = file
        if file.endswith(".dat"):
            data["datfilename"] = file
    data["channels"] = channels
    data["error"] = error
    return render_template('/upload.html',data=data)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_comtrade():
    if request.method == 'POST':
        vr = request.form['vr']
        vs = request.form['vs']
        vt = request.form['vt']
        channels = [int(vr),int(vs),int(vt)]
        print(channels)
        # return ca.analyze(channels)
        # return ','.join()
        # print(ca.standard_analyze(channels))
        result = ca.standard_analyze(channels)
        data = {}
        if len(result) == 3:
            data["result"] = None
            data["result_VR"] = result[0]
            data["result_VS"] = result[1]
            data["result_VT"] = result[2]
        else:
            result = ', '.join(result)
            data["result"] = result
        data["cfgfilename"] = None
        data["datfilename"] = None
        dir = UPLOAD_FOLDER
        for file in os.listdir(dir):
            if file.endswith(".cfg"):
                data["cfgfilename"] = file
            if file.endswith(".dat"):
                data["datfilename"] = file
        return render_template('/result.html', data=data)
    else:
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    # app.run()
    app.run(debug = False)
