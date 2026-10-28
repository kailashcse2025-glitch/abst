import math
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# -------------------------------------------------------
# Confidence & Expiry Math
# -------------------------------------------------------
def confidence(t_now, t0, h):
    """Exponential decay confidence score in [0, 1]."""
    if h <= 0:
        return 1.0 if t_now <= t0 else 0.0
    return math.pow(2, -(t_now - t0) / h)

def expiry_time(t0, h, threshold):
    """Time at which confidence drops to threshold."""
    if threshold >= 1:
        return t0
    return  t0- h * (math.log2(threshold))

# -------------------------------------------------------
# Record & Node Classes
# -------------------------------------------------------
class TrainRecord:
    def __init__(self, train_id, position_km, t0, h, speed_km_per_min=0.5):
        self.train_id = train_id
        self.base_position_km = position_km  # position at t0
        self.t0 = t0                         # last confirmed time
        self.h = h                           # half life
        self.speed = speed_km_per_min        # km per minute travel speed

    def position_at(self, t_now):
        """Calculate dynamic position based on elapsed time since last ping."""
        elapsed = max(0, t_now - self.t0)
        return self.base_position_km + (self.speed * elapsed)

    def confidence_at(self, t_now):
        return confidence(t_now, self.t0, self.h)

class Node:
    def __init__(self, record):
        self.key = record.train_id
        self.record = record
        self.left = None
        self.right = None
        self.t0_max = 0.0
        self.h_max = 0.0
        self.t0_min = 0.0
        self.h_min = 0.0
        self.min_expiry_half = 0.0
        self.refresh()

    def refresh(self):
        t0_vals = [self.record.t0]
        h_vals = [self.record.h]
        expiries = [self.record.t0 + self.record.h]

        for child in [self.left, self.right]:
            if child is not None:
                t0_vals.extend([child.t0_max, child.t0_min])
                h_vals.extend([child.h_max, child.h_min])
                expiries.append(child.min_expiry_half)

        self.t0_max = max(t0_vals)
        self.t0_min = min(t0_vals)
        self.h_max = max(h_vals)
        self.h_min = min(h_vals)
        self.min_expiry_half = min(expiries)

# -------------------------------------------------------
# Decay Tree Implementation
# -------------------------------------------------------
class DecayTree:
    def __init__(self):
        self.root = None
        self.index_by_train = {}

    def insert(self, record):
        self.root = self._insert(self.root, record)

    def _insert(self, node, record):
        if node is None:
            new_node = Node(record)
            self.index_by_train[record.train_id] = new_node
            return new_node
        
        if record.train_id < node.key:
            node.left = self._insert(node.left, record)
        elif record.train_id > node.key:
            node.right = self._insert(node.right, record)
        else:
            node.record = record
            self.index_by_train[record.train_id] = node
            
        node.refresh()
        return node

    def corroborate(self, train_id, t_now, new_position_km=None):
        if train_id not in self.index_by_train:
            raise KeyError(f"Unknown train: {train_id}")
        
        node = self.index_by_train[train_id]
        # If a manual new position is provided, set it; otherwise project position at t_now
        if new_position_km is not None:
            node.record.base_position_km = new_position_km
        else:
            node.record.base_position_km = node.record.position_at(t_now)
            
        node.record.t0 = t_now
        self._rebuild_path(train_id)

    def _rebuild_path(self, train_id):
        path = []
        node = self.root
        while node:
            path.append(node)
            if train_id < node.key:
                node = node.left
            elif train_id > node.key:
                node = node.right
            else:
                break
        for n in reversed(path):
            n.refresh()

    def query_above(self, t_now, min_confidence):
        results = []
        self._query(self.root, t_now, min_confidence, results)
        return results

    def _query(self, node, t_now, threshold, results):
        if node is None:
            return

        upper = confidence(t_now, node.t0_max, node.h_max)
        if upper < threshold:
            return

        lower = confidence(t_now, node.t0_min, node.h_min)
        if lower >= threshold:
            self._collect_all(node, t_now, results)
            return

        c = node.record.confidence_at(t_now)
        if c >= threshold:
            results.append({
                "trainId": node.record.train_id,
                "positionKm": node.record.position_at(t_now),
                "confidence": c
            })

        self._query(node.left, t_now, threshold, results)
        self._query(node.right, t_now, threshold, results)

    def _collect_all(self, node, t_now, results):
        if node is None:
            return
        results.append({
            "trainId": node.record.train_id,
            "positionKm": node.record.position_at(t_now),
            "confidence": node.record.confidence_at(t_now)
        })
        self._collect_all(node.left, t_now, results)
        self._collect_all(node.right, t_now, results)

    def next_expiry(self, threshold=0.5):
        if self.root is None:
            return None
        candidates = []
        self._collect_expiries(self.root, threshold, candidates)
        if not candidates:
            return None
        return min(candidates, key=lambda x: x["time"])

    def _collect_expiries(self, node, threshold, out):
        if node is None:
            return
        exp_time = expiry_time(node.record.t0, node.record.h, threshold)
        out.append({"trainId": node.record.train_id, "time": exp_time})
        self._collect_expiries(node.left, threshold, out)
        self._collect_expiries(node.right, threshold, out)

# -------------------------------------------------------
# Function to Initialize / Reset Tree with Speeds
# -------------------------------------------------------
def create_fresh_tree():
    new_tree = DecayTree()
    # Train A starts at 10km, moving at 0.4 km/min, half-life 20 min
    new_tree.insert(TrainRecord("A", 10.0, 0, 20, speed_km_per_min=0.4))
    # Train B starts at 25km, moving at 0.7 km/min, half-life 10 min
    new_tree.insert(TrainRecord("B", 25.0, 0, 10, speed_km_per_min=0.7))
    # Train C starts at 5.0km, moving at 0.3 km/min, half-life 40 min
    new_tree.insert(TrainRecord("C", 5.0, 0, 40, speed_km_per_min=0.3))
    return new_tree

tree = create_fresh_tree()

# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    global tree
    tree = create_fresh_tree()
    return jsonify({"success": True})

@app.route("/api/state", methods=["GET"])
def get_state():
    t_now = float(request.args.get("tNow", 0))
    threshold = float(request.args.get("threshold", 0.6))

    all_trains = []
    for tid, node in sorted(tree.index_by_train.items()):
        rec = node.record
        conf = rec.confidence_at(t_now)
        pos = rec.position_at(t_now)
        all_trains.append({
            "trainId": rec.train_id,
            "positionKm": pos,
            "t0": rec.t0,
            "h": rec.h,
            "speed": rec.speed,
            "confidence": conf,
            "isTrusted": conf >= threshold
        })

    trusted_trains = tree.query_above(t_now, threshold)
    next_exp = tree.next_expiry(0.5)

    return jsonify({
        "tNow": t_now,
        "threshold": threshold,
        "allTrains": all_trains,
        "trustedTrains": trusted_trains,
        "nextExpiry": next_exp
    })

@app.route("/api/ping", methods=["POST"])
def corroborate_train():
    data = request.json
    train_id = data.get("trainId")
    t_now = float(data.get("tNow", 0))
    new_pos = data.get("positionKm")
    
    if new_pos is not None:
        new_pos = float(new_pos)

    try:
        tree.corroborate(train_id, t_now, new_pos)
        return jsonify({"success": True})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)