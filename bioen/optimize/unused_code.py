from __future__ import print_function


def get_dim_file(file_name):
    """
    Returns number of N structures and M observables from a file

    Parameters
    ----------
    file_name: file containing bioen data

    Returns
    -------
    M: observables
    M: structures

    """

    # yTilde is always allocated on the 4th position
    # of forces and log_weights files and it has
    # dimensionality MxN.

    with open(file_name, 'r') as ifile:
        pack = pickle.load(ifile)

    yTilde = pack[3]
    M = yTilde.shape[0]
    N = yTilde.shape[1]
    return M, N


def load_forces(file_name):
    """
    Extracts bioen data for forces method from a file

    Parameters
    ----------
    file_name: file containing bioen data for forces

    Returns
    -------
    forces:
    w0:
    y:
    yTilde:
    YTilde:
    theta:

    """

    # forces, M x 1
    # w0    , N x 1
    # y     , M x N
    # yTilde, M x N
    # YTilde, 1 x M
    with open(file_name, 'r') as ifile:
        pack = pickle.load(ifile)

    if len(pack) != 6:
        print("Error loading forces data from file ", file_name)
        sys.exit(0)

    forces = pack[0]
    w0 = pack[1]
    y = pack[2]
    yTilde = pack[3]
    YTilde = pack[4]
    theta = pack[5]

    M = yTilde.shape[0]
    N = yTilde.shape[1]

    # forces: Mx1
    currx = w0.shape[0]
    curry = w0.shape[1]
    if (currx != M and curry != 1):
        print("Error on dimensionality of forces(", M, "x1) : but -->", currx, "x", curry)
        sys.exit(0)

    # w0: Nx1
    currx = w0.shape[0]
    curry = w0.shape[1]
    if (currx != N and curry != 1):
        print("Error on dimensionality of w0(", N, "x1) : but -->", currx, "x", curry)
        sys.exit(0)

    # y: MxN
    currx = y.shape[0]
    curry = y.shape[1]
    if (currx != M and curry != N):
        print("Error on dimensionality of y(", M, "x", N, ") : but -->", currx, "x", curry)
        sys.exit(0)

    # YTilde: 1xN
    currx = YTilde.shape[0]
    curry = YTilde.shape[1]
    if (currx != 1 and curry != M):
        print("Error on dimensionality of YTilde(1x", M, ") : but -->", currx, "x", curry)
        sys.exit(0)

    return forces, w0, y, yTilde, YTilde, theta


def load_logw(file_name):
    """
    Extracts bioen data for log_weights method from a file

    Parameters
    ----------
    file_name: file containing bioen data for log_weights

    Returns
    -------
    GInit:
    G:
    y:
    yTilde:
    YTilde:
    theta:

    """
    # GInit , N x 1
    # G     , N x 1
    # y     , M x N
    # yTilde, M x N
    # YTilde, 1 x M
    # w0    , N x 1
    # theta

    with open(file_name, 'r') as ifile:
        pack = pickle.load(ifile)

    if len(pack) != 7:
        print("Error loading log_weights data from file ", file_name)
        sys.exit(0)

    GInit = pack[0]
    G = pack[1]
    y = pack[2]
    yTilde = pack[3]
    YTilde = pack[4]
    w0 = pack[5]
    theta = pack[6]

    M = yTilde.shape[0]
    N = yTilde.shape[1]

    # GInit:  Nx1
    currx = GInit.shape[0]
    curry = GInit.shape[1]
    if (currx != N and curry != 1):
        print("Error on dimensionality of GInit(", N, "x1) : but -->", currx, "x", curry)
        sys.exit(0)

    # G: Nx1
    currx = G.shape[0]
    curry = G.shape[1]
    if (currx != N and curry != 1):
        print("Error on dimensionality of G(", N, "x1) : but -->", currx, "x", curry)
        sys.exit(0)

    # y: MxN
    currx = y.shape[0]
    curry = y.shape[1]
    if (currx != M and curry != N):
        print("Error on dimensionality of y(", M, "x", N, ") : but -->", currx, "x", curry)
        sys.exit(0)

    # YTilde: 1xN
    currx = YTilde.shape[0]
    curry = YTilde.shape[1]
    if (currx != 1 and curry != M):
        print("Error on dimensionality of YTilde(1x", M, ") : but -->", currx, "x", curry)
        sys.exit(0)

    # w0: Nx1
    currx = w0.shape[0]
    curry = w0.shape[1]
    if (currx != N and curry != 1):
        print("Error on dimensionality of w0(", N, "x1) : but -->", currx, "x", curry)
        sys.exit(0)

    return GInit, G, y, yTilde, YTilde, theta


def load_new_config_yaml(file_name, minimizer, packed_params):
    """
    Updates the information in packed_params (coming from a yaml template)
    using the data obtained from a user yaml file containing a specific
    configuration for the given minimizer.
    It is used to safely overwrite data from a user yaml file that can have
    incomplete definition of all the parameters.


    Parameters
    ----------
    XX: array of length XXXXX
    file_name: user's yaml configuration file
    minimizer: specific section to be loaded
    packed_params: map structure containing initial configuration

    Returns
    -------
    packed_params: map structure with updated configuration

    """

    with open(file_name, "r") as fp:
        cfg = yaml.load(fp)

    if "general" in cfg.keys():
        for entry in cfg["general"]:
            packed_params[entry] = cfg["general"][entry]

    if "c_functions" in cfg.keys():
        for entry in cfg["c_functions"]:
            packed_params[entry] = cfg["c_functions"][entry]

    if minimizer in cfg.keys():
        for entry in cfg[minimizer]:
            packed_params["params"][entry] = cfg[minimizer][entry]

    params = packed_params["params"]

    if "algorithm" in params.keys():
        algorithm = params["algorithm"]
        del params["algorithm"]
    else:
        algorithm = ""

    # update minimizer name for scipy
    if minimizer.upper() == "SCIPY":
        key_minimizer = "scipy/c"
    else:
        key_minimizer = minimizer

    if "use_c_functions" in params.keys():
        if params["use_c_functions"] == False:
            if minimizer.upper() == "SCIPY":
                key_minimizer = "scipy/py"
        del params["use_c_functions"]

    # old name
    packed_params["key_minimizer"] = minimizer
    # new name
    minimizer = key_minimizer
    packed_params["minimizer"] = key_minimizer
    packed_params["algorithm"] = algorithm

    return packed_params
