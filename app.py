import os
import threading
from datetime import datetime
from typing import Tuple, List

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import func, or_, not_
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename

from handler import handle_patient, check_file
from models import Task, Patient, SessionManager, metadata, engine

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

with SessionManager() as session:
    # session.drop_all()
    metadata.create_all(engine)


@app.route('/api/express_check', methods=['POST'])
def express_check():
    with SessionManager() as session:
        task = Task(status='done', type='express')
        session.add(task)
        session.commit()

        patient = handle_patient(task.id, request.json['anamnesis'].strip())
        session.add(patient)

        return jsonify({
            'patient': patient.to_json(session)
        }), 200


@app.route('/api/tasks', methods=['POST'])
def create_tasks():
    if 'file' not in request.files or not request.files['file'] or not request.files['file'].filename:
        print('\'file\' not found in request.files')
        return jsonify({
            'message': '\'file\' not found in request.files'
        }), 400
    file = request.files['file']
    file_ext = file.filename.split('.')[1]
    if file_ext != 'csv':
        print(f"bad ext {file_ext!r}")
        return jsonify({
            'message': f"Bad ext {file_ext!r}"
        }), 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filename)
    print(f'File saved as {filename}')

    with SessionManager() as session:
        task = Task(status='run', type='full')
        session.add(task)
        session.commit()

        thread = threading.Thread(None, target=check_file, args=(filename, task.id))
        thread.start()

        return jsonify({
            'task': task.to_json(session)
        }), 200


@app.route('/api/tasks', methods=['GET'], defaults={'task_id': None})
@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_tasks(task_id):
    with SessionManager() as session:
        if task_id == 0:
            task = session.query(Task).order_by(Task.id.desc()).first()
            return jsonify({
                'task': task.to_json(session) if task else None
            }), 200

        if task_id:
            query = session.query(Patient).filter(Patient.task_id == task_id)
            count_tasks = None
            is_dangerous = request.args.get('isDangerous', default=None, type=lambda v: v.lower() == 'true')
            if is_dangerous is not None:
                cond = or_(Patient.selected_result == 1, Patient.predicted_result == 1)
                if not is_dangerous:
                    cond = not_(cond)
                query = query.filter(cond)
            query = query.order_by(Patient.id)
        else:
            count_tasks = session.query(func.count(Task.id)).scalar()
            query = session.query(Task)
            task_type = request.args.get('type', default=None, type=str)
            if task_type:
                query = query.filter(Task.type == task_type)
                count_tasks = session.query(func.count(Task.id)).filter(Task.type == task_type).scalar()
            query = query.order_by(Task.id.desc())

        if 'take' in request.args:
            query = query.limit(request.args['take'])
        if 'skip' in request.args:
            query = query.offset(request.args['skip'])

        return jsonify({
            'count': count_tasks,
            'patients' if task_id else 'tasks': [task.to_json(session) for task in query.all()]
        }), 200

print(os.listdir('.'))

@app.route("/", defaults={'_': ''})
@app.route("/<path:_>")
def main_route(_):
    return render_template("index.html")
