from flask import Flask,send_file,request
from service import start_clustering
import time

app = Flask(__name__)
# Where the main server stores the client's encrypted query feature vector.
# The cluster reads this same path off the shared volume.
SHARE_PATH = '/workspace/shared_data/ciphertexts/target_enc'

@app.route("/blindtouch", methods = ['POST'])
def auth_api():
    f = request.files['target_enc']
    f.save(SHARE_PATH)
    START_TIME = time.time()
    RESULT_PATH = start_clustering()
    return send_file(RESULT_PATH,as_attachment=True)

## Set the port (Default is 8090)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8090")
