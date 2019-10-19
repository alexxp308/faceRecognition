import json

class GenericResult:
    success = False
    status = 200
    message = ""
    resultMapping = {}
    def __init__(self, success, status, message, resultMapping):
        self.success = success
        self.status = status
        self.message = message
        self.resultMapping = resultMapping
