#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
class ResponseModel:
   code=0
   message='success'
   result=''

 
   def __init__(self,code,message,result):
      self.code = code
      self.message = message
      self.result = result  
 