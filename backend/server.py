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

def save_log_to_firestore(username: str, session_id: str, doc_name: str, log_data: Dict[str, Any]):
    """
    Save to:
    users/{username}/sessions/{session_id}/{doc_name}
    """
    try:
        doc_ref = (
            db.collection("users")
            .document(username)
            .collection("sessions")
            .document(session_id)
            .collection("logs")
            .document(doc_name)
        )
        doc_ref.set(log_data)
        print(f"[FIRESTORE] Saved {doc_name} for {username}/{session_id}")
    except Exception as e:
        print(f"[ERROR] Firestore save failed: {e}")

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
@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    username = data.get('username')

    if not username:
        return jsonify({"message": "Username is required."}), 400

    # create a unique session ID
    session_id = f"sess_{uuid.uuid4().hex}"
    
    sessions[session_id] = {
        "username": username,
        "attempt_count": 0,
        "start_time": time.time(),
        "status": "active",
        "queries": [],
        "session_metrics_history": [] 
    }

    log_entry = {
        "log_type": "SESSION_START",
        "session_id": session_id,
        "username": username,
        "start_time": time.time()
    }
    save_log_to_firestore(username, session_id, "session_start", log_entry)

    print(f"[START] New session created for {username}: {session_id}")
    return jsonify({"session_id": session_id})


@app.route('/submit_query', methods=['POST'])
def submit_query():
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('query')
    events = data.get('events', []) 
    
    session = sessions.get(session_id)
    if not session or session['status'] != 'active':
        return jsonify({"message": "Session not found or inactive."}), 404
    
    if not query:
        return jsonify({"message": "Query is empty."}), 400
    
    # calculate typing metrics from events
    typing_metrics = calculate_typing_metrics(events)
    
    # update session attempt count
    session['attempt_count'] += 1
    current_attempt_number = session['attempt_count']
    history: List[Dict[str, Any]] = session.get('session_metrics_history', [])
    
    # store metrics in session history
    attempt_metrics_to_store = {
        "avg_dwell_time_ms": typing_metrics.get("avg_dwell_time_ms", 0.0),
        "avg_flight_time_ms": typing_metrics.get("avg_flight_time_ms", 0.0),
        "keys_per_sec": typing_metrics.get("keys_per_sec", 0.0),
        "backspace_rate": typing_metrics.get("backspace_rate", 0.0),
        "delete_rate": typing_metrics.get("delete_rate", 0.0),
        "total_duration_seconds": typing_metrics.get("total_duration_seconds", 0.0) 
    }
    history.append(attempt_metrics_to_store)

    user_profile = {
        "typing_speed": typing_metrics.get("keys_per_sec", 0.0),
        "avg_flight_time": typing_metrics.get("avg_flight_time_ms", 0.0),
        "avg_dwell_time": typing_metrics.get("avg_dwell_time_ms", 0.0),
        "backspace_rate": typing_metrics.get("backspace_rate", 0.0),
        "delete_rate": typing_metrics.get("delete_rate", 0.0),
        "retry_count": current_attempt_number - 1, 
    }

    # use LLMs to generate adaptive feedback
    llm_response_json_string_raw = generate_sql_feedback(query, user_profile)
    cleaned_json_string = llm_response_json_string_raw.strip()
    if cleaned_json_string.startswith("```json"):
        cleaned_json_string = cleaned_json_string.removeprefix("```json").removesuffix("```").strip()

    try:
        llm_feedback_data = json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON Decode Error: {e}")
        print(f"Failed to decode: {cleaned_json_string}")
        return jsonify({"message": "LLM response is not valid JSON after cleaning.", "details": str(e)}), 500
    
    attempt_log = {
        "log_type": "ATTEMPT",
        "timestamp": time.time(),
        "session_id": session_id,
        "username": session['username'],
        "attempt_number": current_attempt_number,
        "retry_count": current_attempt_number - 1,
        "query": query,
        "events_logged": len(events),
        **typing_metrics, 
        "llm_feedback": llm_feedback_data
    }
    save_log_to_firestore(session["username"], session_id, f"attempt_{current_attempt_number}", attempt_log)

    print(f"[SUBMIT] Session {session_id} - Attempt {current_attempt_number}. Feedback: {llm_feedback_data.get('error_type', 'N/A')}")
    
    return jsonify(llm_feedback_data)

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
    total_attempts = session.get('attempt_count', 0)
    
    # calculate session-level average metrics
    history: List[Dict[str, Any]] = session.get('session_metrics_history', [])
    session_averages: Dict[str, Any] = {}
    final_cluster_id = -1 
    
    if history:
        sum_metrics = {
            "avg_dwell_time_ms": 0.0, 
            "avg_flight_time_ms": 0.0,
            "keys_per_sec": 0.0,
            "backspace_rate": 0.0,
            "delete_rate": 0.0,
        }

        for attempt in history:
            for key in sum_metrics.keys():
                sum_metrics[key] += attempt.get(key, 0.0)
          
        num_attempts = len(history)
        
        if num_attempts > 0:
            metrics_for_cluster = {}
            
            for key, total_sum in sum_metrics.items():
                average_value = total_sum / num_attempts
                session_averages[f'avg_session_{key}'] = round(average_value, 4)
                if key == 'keys_per_sec':
                    metric_name = 'typing_speed'
                else:
                    # avg_dwell_time_ms -> avg_dwell_time
                    metric_name = key.replace('_ms', '') 

                metrics_for_cluster[metric_name] = average_value
            
            metrics_for_cluster["retry_count"] = total_attempts - 1 
            
            # cluster prediction
            try:
                final_cluster_id = predict_cluster(metrics_for_cluster)
                session_averages['session_cluster_id'] = final_cluster_id
            except Exception as e:
                print(f"[ERROR] Final Cluster Prediction Failed. Check if predict_cluster is working correctly and features are correct: {e}")
                
    # record end session log
    end_log = {
        "log_type": "END",
        "timestamp": end_time,
        "session_id": session_id,
        "username": session['username'],
        "total_attempts": total_attempts,
        "message": "Session ended successfully",
        **session_averages 
    }
    save_log_to_firestore(session["username"], session_id, "session_end", end_log)

    
    print(f"[END] Session {session_id} ended. Duration: {duration:.2f}s. Final Cluster: {final_cluster_id}")
    
    del sessions[session_id]

    return jsonify({"status": "ok", "message": "Session ended successfully.", "final_cluster_id": final_cluster_id})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
