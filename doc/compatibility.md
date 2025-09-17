Tested with:

- OS:
    - Ubuntu 22.04 / 24.04
    - MacOS Sequoia (15.4.1)
    - Windows + WSL Ubuntu 22.04
- Python:
    - 3.9: issue with OpenSSL
    - 3.10: CoReSi not compatible
    - 3.11: OK
    - 3.12: OK
    - 3.13: opengate-core not compatible yet
- OpenGate:
    - 10.0.0, 10.0.1
    - 10.0.2: problem with opengate/Allpix interface, see TODOs
- Allpix²:
    - 3.1.0, 3.2.0
- GPU:
    - GeForce RTX 2080 Ti + cupy-cuda115/128
    - Apple M1

By default, pip installs:

opengate 10.0.0 if Python 3.9, 3.10
opengate 10.0.2 if Python 3.11