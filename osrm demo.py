import requests, json

#we query the http API 
#to do http stuff in python we need requests

#the API wants a specific format in the url to give us the right response
url = "http://router.project-osrm.org/table/v1/bike/"
example_points = [(55.94934, -3.20824), (55.94677,-3.23046)] # somewhere in edinburgh

#its longitude first  <------------------
coords_format = ";".join( f"{lon},{lat}" for lat, lon in example_points )
url += coords_format
url += "?annotations=distance"
print(url)

#send off a request and get back a response
r = requests.get(url)
#the response is in JSON so handle it as such
r = json.loads(r.content)
# it gives us a lot of other stuff but we only need the distance matrix
print(r["distances"])
