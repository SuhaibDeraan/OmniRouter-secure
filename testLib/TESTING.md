# Guide to testing

chat_model -> tests all chat models
image_model -> tests all image models
test_user -> tests a single model of your choice

# Required environment

The suite makes authenticated calls against a running router, so it needs a
valid OmniRouter API key. It is read from the environment and is never committed:

```
TEST_OMNI_API_KEY   # a valid OmniRouter API key (Bearer token for the test user)
```

If `TEST_OMNI_API_KEY` is unset, the integration tests are skipped rather than
failing. The server itself also needs its usual provider/Firebase environment
variables (see `.env.example`).

# Run all tests
Run: `pytest`


# To test chat models

Run: `python -m pytest testLib/test_chat_model.py -v`


# To test image models

Run: `python -m pytest testLib/test_image.py -v`

# To test a single model

Run: `python -m testLib.test_user model_name`
