from io import StringIO
import sys
import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np
from spacy.en import English

nlp = English()

class OmegleEnv(gym.Env):
    def __init__(self, beam_size=16, goal_reward=10.0):
        super(OmegleEnv, self).__init__()

        self.beam_size = beam_size
        self.embedding_dim = 300
        self._action_space = spaces.Discrete(98)

        self.driver = webdriver.Chrome()

        self.last_message = None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf,
                          shape=(self.beam_size, self.embedding_dim))

    def _close(self):
        self.driver.close()

    def _reset(self):
        try:
            self.msg_box.send_keys(Keys.ESCAPE)
            time.sleep(1)
            self.msg_box.send_keys(Keys.ESCAPE)
            time.sleep(1)
        except:
            pass
        self.driver.get("http://www.omegle.com")
        self.driver.find_element_by_id("textbtn").click()
        time.sleep(1)
        self.msg_box = self.driver.find_element_by_class_name("chatmsg")
    
    def _send_action(self, action):
        if action == 0:
            self.msg_box.send_keys(Keys.RETURN)

        elif action == 1:
            self.msg_box.clear()

        elif action == 2:
            time.sleep(1)

        else:
            self.msg_box.send_keys(chr(action+29))

    def _encode(self, text):
        vecs = []
        if text:
            doc = nlp(text)
            doc = doc[:self.beam_size]
            for w in doc:
                vecs.append(w.vector.reshape((1, -1)))
        else:
            doc = []   
        for i in range(self.beam_size-len(doc)):
            vecs.append(np.zeros((1, self.embedding_dim)))
        return np.concatenate(vecs)

    def _has_finished(self):
        status_els = self.driver.find_elements_by_css_selector(".statuslog")
        return any(map(lambda el: el.text == 'Stranger has disconnected.', status_els))

    def _get_last_message(self):
        els = self.driver.find_elements_by_css_selector(".strangermsg span")
        if len(els):
            return els[-1].text
        
    def _observe(self):
        return self._encode(self._get_last_message())

    def _reward(self, action):
        curr_message = self._get_last_message()
        if curr_message != self.last_message:
            self.last_message = curr_message
            return 1
        else:
            return 0

    def _step(self, action):
        reward = self._reward(action)
        
        self._send_action(action)
        time.sleep(0.1)

        obs = self._observe()
        has_finished = self._has_finished()
        info = {}

        return obs, reward, has_finished, info
