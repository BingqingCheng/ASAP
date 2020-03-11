import unittest
import os

import asaplib
print('Successfully imported asaplib')

# find tests and run them
suite = unittest.defaultTestLoader.discover(os.getcwd())
unittest.TextTestRunner().run(suite)
