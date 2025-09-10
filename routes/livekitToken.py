# # server.py
# import os
# # from livekit import api
# from fastapi import APIRouter
# from dotenv import load_dotenv

# load_dotenv()
# router = APIRouter()

# # @router.get("/getToken/")
# # def getToken():
# #   token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
# #     .with_identity("identity") \
# #     .with_name("my name") \
# #     .with_grants(api.VideoGrants(
# #         room_join=True,
# #         room="insurance",
# #     ))
# #   return token.to_jwt()