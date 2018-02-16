import idigbio
api = idigbio.json()
record_list = api.search_records(rq={"scientificname": "Tabebuia impetiginosa", "hasImage": True})
print(record_list)
