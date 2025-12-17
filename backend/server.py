from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import time
import json 
from helper.adaptive_feedback_pipeline_v2 import generate_sql_feedback
from helper.user_tracking import calculate_typing_metrics
from helper.predict_cluster import predict_cluster
import os
from typing import Dict, Any, List
import json
import firebase_admin
from firebase_admin import credentials, firestore


# initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://adaptive-sql-learning.vercel.app"
]}})


# initialize Firestore
cred = credentials.Certificate(json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]))
firebase_admin.initialize_app(cred)
db = firestore.client()

# store sessions in memory 
sessions: Dict[str, Dict[str, Any]] = {}

def save_log_to_firestore(username, session_id, task_id, attempt_number, log_data):
    """
    Save to:
    users/{username}/sessions/{session_id}/tasks/{task_id}/attempts/{attempt_number}
    """
    try:
        doc_ref = (
            db.collection("users")
            .document(username)
            .collection("sessions")
            .document(session_id)
            .collection("tasks")
            .document(str(task_id))
            .collection("attempts")
            .document(str(attempt_number))
        )
        doc_ref.set(log_data)
        print(f"[FIRESTORE] Saved: users/{username}/sessions/{session_id}/tasks/{task_id}/attempts/{attempt_number}")
    except Exception as e:
        print(f"[ERROR] Firestore save failed: {e}")

def save_session_end_to_firestore(username, session_id, task_id, log_data):
    """
    Save session end log to:
    users/{username}/sessions/{session_id}/tasks/{task_id}/attempts/session_end
    """
    try:
        doc_ref = (
            db.collection("users")
            .document(username)
            .collection("sessions")
            .document(session_id)
            .collection("tasks")
            .document(str(task_id))
            .collection("attempts")
            .document("session_end")
        )
        doc_ref.set(log_data)
        print(f"[FIRESTORE] Saved session end: users/{username}/sessions/{session_id}/tasks/{task_id}/attempts/session_end")
    except Exception as e:
        print(f"[ERROR] Firestore session end save failed: {e}")

# def write_log_line(log_data: Dict[str, Any]):
#     try:
#         username = log_data.get('username')
#         session_id = log_data.get('session_id')
        
#         if not username or not session_id:
#             print("[ERROR] Log data missing username or session_id, skipping logging.")
#             return

#         # create user-specific log directory if not exists
#         user_log_dir = os.path.join(LOG_DIR, username)
#         if not os.path.exists(user_log_dir):
#             os.makedirs(user_log_dir)
            
#         # log file path
#         log_file_path = os.path.join(user_log_dir, f'{session_id}.jsonl')
            
#         log_json_line = json.dumps(log_data, ensure_ascii=False)
#         with open(log_file_path, 'a', encoding='utf-8') as f:
#             f.write(log_json_line + '\n')
            
#     except Exception as e:
#         print(f"[ERROR] Failed to write log line: {e}")

# define routes and logic
@app.route('/start', methods=['POST'])
def start_session():
    data = request.get_json()
    username = data.get('username')
    
    if not username:
        return jsonify({"message": "Username is required."}), 400
    
    # Create unique session ID
    session_id = f"sess_{uuid.uuid4().hex}"
    
    # Initialize session in memory
    sessions[session_id] = {
        "username": username,
        "attempt_count": 0,
        "start_time": time.time(),
        "status": "active",
        "tasks": {},
    }
    
    # Create/update user document in Firestore
    try:
        user_ref = db.collection('users').document(username)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            user_ref.set({
                'username': username,
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_login': firestore.SERVER_TIMESTAMP
            })
        else:
            user_ref.update({
                'last_login': firestore.SERVER_TIMESTAMP
            })
    except Exception as e:
        print(f"[ERROR] Failed to create/update user: {e}")
    
    print(f"[START] New session created for {username}: {session_id}")
    return jsonify({"session_id": session_id})

@app.route('/submit_query', methods=['POST'])
def submit_query():
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('query')
    events = data.get('events', [])
    is_correct = data.get('is_correct', False)
    task_id = data.get('task_id')

    if not session_id or task_id is None:
        return jsonify({"message": "Missing required fields."}), 400
    
    session = sessions.get(session_id)
    if not session or session['status'] != 'active':
        return jsonify({"message": "Session not found or inactive."}), 404

    if not query:
        return jsonify({"message": "Query is empty."}), 400

    # Initialize task tracking if first attempt
    if "tasks" not in session:
        session["tasks"] = {}
    if task_id not in session["tasks"]:
        session["tasks"][task_id] = {
            "attempts": 0,
            "metrics_history": [] 
        }
    
    # Calculate typing metrics from events
    typing_metrics = calculate_typing_metrics(events)
    
    # Update attempt counts
    session["tasks"][task_id]["attempts"] += 1
    current_attempt_number = session["tasks"][task_id]["attempts"]
 
    session["tasks"][task_id]["metrics_history"].append({
        "attempt_number": current_attempt_number,
        "is_correct": is_correct,
        "avg_dwell_time_ms": typing_metrics.get("avg_dwell_time_ms", 0.0),
        "avg_flight_time_ms": typing_metrics.get("avg_flight_time_ms", 0.0),
        "keys_per_sec": typing_metrics.get("keys_per_sec", 0.0),
        "backspace_rate": typing_metrics.get("backspace_rate", 0.0),
        "delete_rate": typing_metrics.get("delete_rate", 0.0),
        "total_duration_seconds": typing_metrics.get("total_duration_seconds", 0.0)
    })

    response_data = {}
    llm_feedback_data = None

    if is_correct:
        response_data = {
            "is_correct": True,
            "message": "Correct answer!"
        }

    else:
        # use LLM to generate personalized feedback
        try:
            user_profile = {
                "typing_speed": typing_metrics.get("keys_per_sec", 0.0),
                "avg_flight_time": typing_metrics.get("avg_flight_time_ms", 0.0),
                "avg_dwell_time": typing_metrics.get("avg_dwell_time_ms", 0.0),
                "backspace_rate": typing_metrics.get("backspace_rate", 0.0),
                "delete_rate": typing_metrics.get("delete_rate", 0.0),
                "retry_count": current_attempt_number - 1,
            }

            llm_response_json_string_raw = generate_sql_feedback(query, user_profile)
            cleaned_json_string = llm_response_json_string_raw.strip()
            
            if cleaned_json_string.startswith("```json"):
                cleaned_json_string = cleaned_json_string.removeprefix("```json").removesuffix("```").strip()

            llm_feedback_data = json.loads(cleaned_json_string)
            
            response_data = {
                "is_correct": False,
                "error_type": llm_feedback_data.get("error_type", "UNKNOWN"),
                "error_subtype": llm_feedback_data.get("error_subtype", "UNKNOWN"),
                "personalized_feedback": llm_feedback_data.get("personalized_feedback", "Please try again.")
            }
        except Exception as e:
            print(f"[ERROR] LLM feedback generation failed: {e}")
            response_data = {
                "is_correct": False,
                "error_type": "SYSTEM_ERROR",
                "error_subtype": "LLM_ERROR",
                "personalized_feedback": "An error occurred while generating feedback. Please try again."
            }
    
    # Save attempt log to Firestore
    attempt_log = {
        "log_type": "ATTEMPT",
        "timestamp": time.time(),
        "session_id": session_id,
        "username": session['username'],
        "question_id": task_id,
        "attempt_number": current_attempt_number,
        "retry_count": current_attempt_number - 1,
        "query": query,
        "is_correct": is_correct,
        "events_logged": len(events),
        **typing_metrics,
        "llm_feedback": llm_feedback_data,
    }
    save_log_to_firestore(
        session["username"], 
        session_id, 
        task_id, 
        f"attempt_{current_attempt_number}", 
        attempt_log
    )

    print(f"[SUBMIT] Session {session_id} - Task {task_id} - Attempt {current_attempt_number} - Correct: {is_correct}")
    
    return jsonify(response_data)

@app.route('/end_question', methods=['POST'])
def end_question():
    data = request.get_json()
    session_id = data.get('session_id')
    task_id = data.get('task_id')
    reason = data.get('reason', 'unknown')  # 'correct', 'max_attempts', 'quit'
    
    session = sessions.get(session_id)
    if not session or session['status'] != 'active':
        return jsonify({"message": "Session not found or inactive."}), 404
    
    if not task_id:
        return jsonify({"message": "Missing question_id"}), 400
    
    # Get task attempt data
    task_info = session.get("tasks", {}).get(task_id)
    if not task_info:
        return jsonify({"message": "No attempt data found for this question"}), 404
    
    metrics_history = task_info.get("metrics_history", [])
    total_attempts = task_info.get("attempts", 0)
    
    if not metrics_history:
        return jsonify({
            "status": "ok",
            "message": "Question ended but no metrics available",
            "question_cluster_id": None
        })
    
    # Calculate aggregated metrics
    sum_metrics = {
        "avg_dwell_time_ms": 0.0,
        "avg_flight_time_ms": 0.0,
        "keys_per_sec": 0.0,
        "backspace_rate": 0.0,
        "delete_rate": 0.0,
    }
    
    for attempt in metrics_history:
        sum_metrics["avg_dwell_time_ms"] += attempt.get("avg_dwell_time_ms", 0.0)
        sum_metrics["avg_flight_time_ms"] += attempt.get("avg_flight_time_ms", 0.0)
        sum_metrics["keys_per_sec"] += attempt.get("keys_per_sec", 0.0)
        sum_metrics["backspace_rate"] += attempt.get("backspace_rate", 0.0)
        sum_metrics["delete_rate"] += attempt.get("delete_rate", 0.0)
    
    num_attempts = len(metrics_history)
    
    # Calculate average metrics
    avg_metrics = {
        "typing_speed": sum_metrics["keys_per_sec"] / num_attempts,
        "avg_dwell_time": sum_metrics["avg_dwell_time_ms"] / num_attempts,
        "avg_flight_time": sum_metrics["avg_flight_time_ms"] / num_attempts,
        "backspace_rate": sum_metrics["backspace_rate"] / num_attempts,
        "delete_rate": sum_metrics["delete_rate"] / num_attempts,
        "retry_count": total_attempts - 1  
    }
    
    # Predict question cluster
    question_cluster_id = None
    try:
        question_cluster_id = predict_cluster(avg_metrics)
        print(f"[END_QUESTION] Task {task_id} - Reason: {reason} - Cluster: {question_cluster_id}")
        print(f"[END_QUESTION] Aggregated metrics: {avg_metrics}")
    except Exception as e:
        print(f"[ERROR] Cluster prediction failed: {e}")
    
    # Save end question log to Firestore
    end_question_log = {
        "log_type": "END_QUESTION",
        "timestamp": time.time(),
        "session_id": session_id,
        "username": session['username'],
        "question_id": task_id,
        "total_attempts": total_attempts,
        "reason": reason,
        "aggregated_metrics": avg_metrics,
        "learner_type": question_cluster_id,
        "all_attempts": metrics_history  
    }
    
    save_log_to_firestore(
        session["username"],
        session_id,
        task_id,
        f"end_{reason}",
        end_question_log
    )
    
    return jsonify({
        "status": "ok",
        "message": f"Question ended: {reason}",
        "question_cluster_id": question_cluster_id,
        "total_attempts": total_attempts
    })

@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.get_json()
    session_id = data.get('session_id')
    
    session = sessions.get(session_id)
    if not session or session['status'] != 'active':
        return jsonify({"message": "Session not found or already ended."}), 404

    session['status'] = 'ended'
    end_time = time.time()
    start_time = session.get('start_time')
    duration = end_time - start_time if start_time else 0
    
    # Session end log data
    end_log = {
        "log_type": "END_SESSION",
        "timestamp": end_time,
        "session_id": session_id,
        "username": session['username'],
        "duration_seconds": duration,
        "message": "Session ended successfully"
    }
    
    tasks = session.get("tasks", {})
    for task_id in tasks.keys():
        try:
            doc_ref = (
                db.collection("users")
                .document(session["username"])
                .collection("sessions")
                .document(session_id)
                .collection("tasks")
                .document(str(task_id))
                .collection("attempts")
                .document("session_end")
            )
            doc_ref.set(end_log)
        except Exception as e:
            print(f"[ERROR] Failed to save session end for task {task_id}: {e}")
    
    print(f"[END_SESSION] Session {session_id} ended. Duration: {duration:.2f}s")
    
    del sessions[session_id]

    return jsonify({
        "status": "ok", 
        "message": "Session ended successfully."
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
    # print("Starting Flask server on http://127.0.0.1:3001")
    # app.run(port=3001, debug=True)
