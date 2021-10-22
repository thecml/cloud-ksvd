# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import struct
import base64
import time
import asyncio
import numpy as np
import json
import aioredis
import asyncio
import nest_asyncio

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

async def close_redis(redis):
    redis.close()
    await redis.wait_closed()

async def save_to_redis(redis,A,key):
    array_dtype = str(A.dtype)
    l, w = A.shape
    A = A.ravel().tostring()
    valkey = '{0}|{1}#{2}#{3}'.format(int(time.time()), array_dtype, l, w)
    await redis.set(valkey, A)
    await redis.set(key, valkey)

async def load_from_redis(redis, key):
    valkey = await redis.get(key, encoding='utf-8')
    if not valkey:
        return []
    A = await redis.get(valkey)
    array_dtype, l, w = valkey.split('|')[1].split('#')
    return np.fromstring(A, dtype=array_dtype).reshape(int(l), int(w))

async def save_redis():
    redis = await aioredis.create_redis('redis://localhost/0')
    await redis.flushdb()
    
    data = {'resultDictionary': np.asmatrix(np.ones((10,20))),
        'resultSignal': np.asmatrix(np.ones((10,20))),
        'resultStats': {'stat1':1}}
    
    D = data['resultDictionary']
    X = data['resultSignal']
    S = data['resultStats']
    
    print("dimensions = ", np.size(D,0))
    print("dimensions2 = ", np.size(D,1))
    
    podIp = '192.168.1.1'
    redis.lpush('resultIps', podIp)
    await save_to_redis(redis, D, podIp + 'resultDictionary')
    await save_to_redis(redis, X, podIp + 'resultSignal')
    await redis.hmset_dict(podIp + 'resultStats', S)
    await close_redis(redis)
    
    #await redis.set('key',encoded)
    #print(await redis.get('key'))
    
async def load_redis():
    redis = await aioredis.create_redis('redis://localhost/0')
    
    podIpList = []
    while(await redis.llen('resultIps')!=0):
        ip = await redis.rpop('resultIps', encoding='utf-8')
        podIpList.append(ip)

    resultList = []
    for podIp in podIpList:
        D = await load_from_redis(redis, podIp + 'resultDictionary')
        X = await load_from_redis(redis, podIp + 'resultSignal')
        S = await redis.hgetall(podIp + 'resultStats', encoding='utf-8')
        resultList.append({'D': D.tolist(),'X':X.tolist(),'S':S})
    
    await close_redis(redis)
    return json.dumps(resultList)
    
nest_asyncio.apply()
asyncio.get_event_loop().run_until_complete(save_redis())
res = asyncio.get_event_loop().run_until_complete(load_redis())