# BAG (BAG AMS Generator) 3

BAG, a recursive acronym which stands for "BAG AMS Generator", is a fork and successor of
the [BAG\_framework](https://github.com/ucb-art/BAG_framework).

## Setup

First, initialize and retrieve submodules:

    $ git submodule update --init --recursive

Building *pybag* and *cbag* (included as submodules) requires:

- A C++17 compiler (GCC)
- Python 3.7+
- [Boost](https://www.boost.org/)
- [CMake](https://cmake.org/)
- [Catch2](https://github.com/catchorg/Catch2)
- [fmt](https://github.com/fmtlib/fmt)
- [spdlog](https://github.com/gabime/spdlog)

BAG is more useful with the [OpenAccess API](http://www.si2.org/openaccess/).
If you want to build BAG with OpenAccess, set the *OA_INCLUDE_DIR* and
*OA_LINK_DIR* environment variables.

BAG also requires several Python dependencies. Those are documented in
*setup.py*.

To build BAG, run:

    $ cd pybag && ./build.sh

BAG does not need to be installed. Instead, set up a "BAG workspace" with this
repository as a subdirectory named *BAG_framework* and copy the scripts from
*run_scripts/*. Then:

- Set *BAG_FRAMEWORK*, *BAG_PYTHON*, and other necessary environment variables
- Add a *bag_submodules.yaml*
- Attach a BAG technology file
- Run `./setup_submodules.py`
- Start BAG with `./start_bag.sh`

## Licensing

This library is licensed under the Apache-2.0 license.  However, some source files are licensed
under both Apache-2.0 and BSD-3-Clause license, meaning that the user must comply with the
terms and conditions of both licenses.  See [here](LICENSE.BSD-3-Clause) for full text of the
BSD license, and [here](LICENSE.Apache-2.0) for full text of the Apache license.  See individual
files to check if they fall under Apache-2.0, or both Apache-2.0 and BSD-3-Clause.
