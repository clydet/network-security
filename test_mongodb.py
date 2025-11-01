from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
ca = certifi.where()

uri = "mongodb+srv://iamclyde_db_user:pNGkuMR1n51ybfay@cluster0.i85mffo.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))
# client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
