import numpy as np
from seal import *
import asyncio
import aiohttp

SHARE_PATH = '/workspace/shared_data/ciphertexts/target_enc'
CLUSTERING_1 = '/workspace/shared_data/results/clustering_1'

RESULT = '/workspace/shared_data/results/compressed'
PK_PATH = '/workspace/shared_data/keys/public.key'
GK_PATH = '/workspace/shared_data/keys/galois.key'

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 16384
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, 40, 40, 40, 60]))
scale = 2.0**40
context = SEALContext(parms)

ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()
print(f'Number of slots: {slot_count}')

public_key = PublicKey()
public_key.load(context, PK_PATH)
galois_key = GaloisKeys()
galois_key.load(context, GK_PATH)
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(fetch(session, url))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)   

# 3-container demo: only cluster1 exists. The original paper fans out to 3
# clusters here and compresses their partial results with add_many.
urls = ['http://cluster1:8090/blindtouch']



def start_clustering():
    print('start')
    responses = asyncio.run(fetch_all(urls))
    print(responses)
    # Single cluster: its partial result IS the compressed result. (With 3
    # clusters this would be evaluator.add_many([ctxt1, ctxt2, ctxt3]).)
    ctxt1 = Ciphertext()
    ctxt1.load(context, CLUSTERING_1)

    result = ctxt1
    result.save(RESULT)
    return RESULT
