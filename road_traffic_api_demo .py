import requests
import json
import pandas as pd
import time

# url = 'https://roadtraffic.dft.gov.uk/api/average-annual-daily-flow'
# headers = {
# 'Content-Type': 'application/json',
# 'Accept': 'application/json'
# }
# params = {
# 'page[size]': '10',
# 'include':'region',
# 'filter[year]':'2024',
# 'filter[region_id]':'3',
# }


def send_request(url, params=None, return_json=True):
    """
    A wrapper to send GET requests to ``URL`` with optional ``params``.
    This will return a ``Response`` object unless ``return_json=True`` in which case
    a conversion is attempted.
    """
    #slow request rate
    time.sleep(1)
    # idk what these are but they were in the demo
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
    }
    #send request and recieve a response
    resp = requests.request('GET', url, headers=headers, params=params)

    # Gracefully handle errors
    if resp.status_code != 200:

        msg = f"Fuck \t Code:{resp.status_code} "

        if resp.status_code == 400:
            msg+= "\n" + resp.json()["message"]
        
        raise Exception(msg)

    if return_json :
        return resp.json()
    
    return resp
    

def read_data_from_api():
    url = 'https://roadtraffic.dft.gov.uk/api/average-annual-daily-flow'
    params = {
    'page[size]': '100',
    'include':'region',
    'filter[year]':'2024',
    'filter[region_id]':'3',
    }

    params = dict(**{"page[number]":1}, **params)

    # send off a request for page one just to get the chain started
    first_response = send_request(url, params=params, return_json=False)
    next_url = first_response.url

    while next_url is not None:
        print(f"{next_url=}")
        response = send_request(url=next_url)
        next_url = response["next_page_url"]

        # now we read the info from response["data"]
        





if __name__ == "__main__":

    x = read_data_from_api()