import pdb

import boto3, json
from botocore.exceptions import ClientError

import datetime
from time import time
from uuid import uuid4

from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from openai import OpenAI
import google.generativeai as genai


class RefreshableBotoSession:
    """
    Boto Helper class which lets us create a refreshable session so that we can cache the client or resource.

    Usage
    -----
    session = RefreshableBotoSession().refreshable_session()

    client = session.client("s3") # we now can cache this client object without worrying about expiring credentials
    """

    def __init__(
        self,
        region_name: str = None,
        profile_name: str = None,
        sts_arn: str = None,
        session_name: str = None,
        session_ttl: int = 3000
    ):
        """
        Initialize `RefreshableBotoSession`

        Parameters
        ----------
        region_name : str (optional)
            Default region when creating a new connection.

        profile_name : str (optional)
            The name of a profile to use.

        sts_arn : str (optional)
            The role arn to sts before creating a session.

        session_name : str (optional)
            An identifier for the assumed role session. (required when `sts_arn` is given)

        session_ttl : int (optional)
            An integer number to set the TTL for each session. Beyond this session, it will renew the token.
            50 minutes by default which is before the default role expiration of 1 hour
        """

        self.region_name = region_name
        self.profile_name = profile_name
        self.sts_arn = sts_arn
        self.session_name = session_name or uuid4().hex
        self.session_ttl = session_ttl

    def __get_session_credentials(self):
        """
        Get session credentials
        """
        session = Session(region_name=self.region_name, profile_name=self.profile_name)

        # if sts_arn is given, get credential by assuming the given role
        if self.sts_arn:
            sts_client = session.client(service_name="sts", region_name=self.region_name)
            response = sts_client.assume_role(
                RoleArn=self.sts_arn,
                RoleSessionName=self.session_name,
                DurationSeconds=self.session_ttl,
            ).get("Credentials")

            credentials = {
                "access_key": response.get("AccessKeyId"),
                "secret_key": response.get("SecretAccessKey"),
                "token": response.get("SessionToken"),
                "expiry_time": response.get("Expiration").isoformat(),
            }
        else:
            session_credentials = session.get_credentials().get_frozen_credentials()
            credentials = {
                "access_key": session_credentials.access_key,
                "secret_key": session_credentials.secret_key,
                "token": session_credentials.token,
                "expiry_time": datetime.datetime.fromtimestamp(time() + self.session_ttl, tz=datetime.timezone.utc).isoformat(),
            }

        return credentials

    def refreshable_session(self) -> Session:
        """
        Get refreshable boto3 session.
        """
        # Get refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self.__get_session_credentials(),
            refresh_using=self.__get_session_credentials,
            method="sts-assume-role",
        )

        # attach refreshable credentials current session
        session = get_session()
        session._credentials = refreshable_credentials
        session.set_config_variable("region", self.region_name)
        autorefresh_session = Session(botocore_session=session)

        return autorefresh_session

class Prompter(object):
    def __init__(self, model):
        session = RefreshableBotoSession().refreshable_session()

        if model in {'gpt', 'gpt3', 'gpt4'}:
            self.client = OpenAI()
        elif model == 'gflash' or model == 'gpro':
            if model == 'gflash': self.client = genai.GenerativeModel("gemini-1.5-flash")
            elif model == 'gpro': self.client = genai.GenerativeModel("gemini-1.5-pro")
            else: assert False
        else: assert False

        self.model = model


    def prompt(self, query):
        if self.model == 'haiku':
            # Format the request payload using the model's native structure.
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.5,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}],
                    }
                ],
            }

            # Convert the native request to JSON.
            request = json.dumps(native_request)

            try:
                # Invoke the model with the request.
                response = self.client.invoke_model(modelId=self.model_id, body=request)

            except (ClientError, Exception) as e:
                print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
                exit(1)

            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            response_text = model_response["content"][0]["text"]
            return response_text
        elif self.model in {'gpt', 'gpt3', 'gpt4'}:
            if self.model == 'gpt': mod = 'gpt-4o'
            elif self.model  == 'gpt3': mod = 'gpt-3.5-turbo-0125'
            elif self.model == 'gpt4': mod = 'gpt-4-0613'
            else: assert False
            try:
                response = self.client.chat.completions.create(
                    model=mod,
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
            except Exception as e:
                assert 'context_length_exceeded' in str(e), str(e)
                frac = 0.95
                all_done=False
                while frac > 0:
                    print("TRYING TO REDUCE DOC1 to frac", frac)
                    newq = handle_gpt_context_issue(query, frac)
                    try:
                        response = self.client.chat.completions.create(
                            model=mod,
                            messages=[
                                {"role": "user", "content": newq}
                            ]
                        )
                        need_again=False
                    except Exception as e:
                        assert 'context_length_exceeded' in str(e)
                        need_again=True
                        frac -= 0.05
                    if not need_again:
                        all_done=True
                        break
                if not all_done:
                    assert False, 'context length issue'
            choices = response.choices
            assert len(choices) == 1
            return choices[0].message.content
        elif self.model == 'gflash' or self.model == 'gpro':
            response = self.client.generate_content(query)
            return response.text
        else: assert False

def handle_gpt_context_issue(query, frac):
    if '</document1>' in query:
        cutter, cutter2 = '</document1>', '<document1>'
    else: cutter, cutter2 = '</document>', '<document>'
    splt = query.split(cutter)
    assert len(splt) >= 2, str(len(splt))
    if len(splt) > 2:
        splt = [cutter.join(splt[:-1]), splt[-1]]
    assert len(splt) == 2
    splt2 = splt[0].split(cutter2)
    if len(splt2) > 2:
        splt2 = [cutter2.join(splt2[:-1]), splt2[-1]]
    assert len(splt2) == 2
    doc1_orig = splt2[1]
    doc1_tmp = splt2[1][:int(frac*len(doc1_orig))]
    query_new = splt2[0]+cutter2+doc1_tmp+cutter+splt[1]
    return query_new
