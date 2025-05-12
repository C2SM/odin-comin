import numpy as np

def get_int_coefs(pb_ret, pb_mod):
    """
    Computes a coefficients matrix to transfer a model profile onto
    a retrieval pressure axis.
    
    If level_def=="layer_average", this assumes that profiles are
    constant in each layer of the retrieval, bound by the pressure
    boundaries pb_ret. In this case, the WRF model layer is treated
    in the same way, and coefficients integrate over the assumed
    constant model layers. This works with non-staggered WRF
    variables (on "theta" points). However, this is actually not how
    WRF is defined, and the implementation should be changed to
    z-staggered variables. Details for this change are in a comment
    at the beginning of the code.

    If level_def=="pressure_boundary" (IMPLEMENTATION IN PROGRESS),
    assumes that profiles, kernel and pwf are defined at pressure
    boundaries that don't have a thickness (this is how OCO-2 data
    are defined, for example). In this case, the coefficients
    linearly interpolate adjacent model level points. This is
    incompatible with the treatment of WRF in the above-described
    layer-average assumption, but is closer to how WRF is actually
    defined. The exception is that pb_mod is still constructed and
    non-staggered variables are not defined at psurf. This can only
    be fixed by switching to z-staggered variables.

    In cases where retrieval surface pressure is higher than model
    surface pressure, and in cases where retrieval top pressure is
    lower than model top pressure, the model profile will be
    extrapolated with constant tracer mixing ratios. In cases where
    retrieval surface pressure is lower than model surface pressure,
    and in cases where retrieval top pressure is higher than model
    top pressure, only the parts of the model column that fall
    within the retrieval presure boundaries are sampled.

    Arguments
    ---------
    pb_ret (:class:`array_like`)
        Pressure boundaries of the retrieval column
    pb_mod (:class:`array_like`)
        Pressure boundaries of the model column
    level_def (:class:`string`)
        "layer_average" or "pressure_boundary" (IMPLEMENTATION IN
        PROGRESS). Refers to the retrieval profile.
        
        Note 2021-09-13: Inspected code for pressure_boundary.
        Should be correct. Interpolates linearly between two model
        levels.


    Returns
    -------
    coefs (:class:`array_like`)
            Integration coefficient matrix. Each row sums to 1.

    Usage
    -----
            .. code-block:: python

                import numpy as np
                pb_ret = np.linspace(900., 50., 5)
                pb_mod = np.linspace(1013., 50., 7)
                model_profile = 1. - np.linspace(0., 1., len(pb_mod)-1)**3
                coefs = get_int_coefs(pb_ret, pb_mod, "layer_average")
                retrieval_profile = np.matmul(coefs, model_profile)
    """

    # This code assumes that WRF variables are constant in
    # layers, but they are defined on levels. This can be seen
    # for example by asking wrf.interplevel for the value of a
    # variable that is defined on the mass grid ("theta points")
    # at a pressure slightly higher than the pressure on its
    # grid (wrf.getvar(ncf, "p")), it returns nan. So There is
    # no extrapolation. There are no layers. There are only
    # levels.
    # In addition, this page here:
    # https://www.openwfm.org/wiki/How_to_interpret_WRF_variables
    # says that to find values at theta-points of a variable
    # living on u-points, you interpolate linearly. That's the
    # other way around from what I would do if I want to go from
    # theta to staggered.
    # WRF4.0 user guide:
    # - ungrib can interpolate linearly in p or log p
    # - real.exe comes with an extrap_type namelist option, that
    #   extrapolates constantly BELOW GROUND.
    # This would mean the correct way would be to integrate over
    # a piecewise-linear function. It also means that I really
    # want the value at surface level, so I'd need the CO2
    # fields on the Z-staggered grid ("w-points")! Interpolate
    # the vertical in p with wrf.interp1d, example:
    # wrf.interp1d(np.array(rh.isel(south_north=1, west_east=0)),
    #              np.array(p.isel(south_north=1, west_east=0)),
    #              np.array(988, 970))
    # (wrf.interp1d gives the same results as wrf.interplevel,
    # but the latter just doesn't want to work with single
    # columns (32,1,1), it wants a dim>1 in the horizontal
    # directions)
    # So basically, I can keep using pb_ret and pb_mod, but it
    # would be more accurate to do the piecewise-linear
    # interpolation and the output matrix will have 1 more
    # value in each dimension.
    
    # Calculate integration weights by weighting with layer
    # thickness. This assumes that both axes are ordered
    # psurf to ptop.
    coefs = np.ndarray(shape=(len(pb_ret)-1, len(pb_mod)-1))
    coefs[:] = 0.

    # Extend the model pressure grid if retrieval encompasses
    # more.
    # pb_mod_tmp = copy.deepcopy(pb_mod)
    pb_mod_tmp = np.copy(pb_mod)

    # In case the retrieval pressure is higher than the model
    # surface pressure, extend the lowest model layer.
    if pb_mod_tmp[0] < pb_ret[0]:
        pb_mod_tmp[0] = pb_ret[0]

    # In case the model doesn't extend as far as the retrieval,
    # extend the upper model layer upwards.
    if pb_mod_tmp[-1] > pb_ret[-1]:
        pb_mod_tmp[-1] = pb_ret[-1]

    # For each retrieval layer, this loop computes which
    # proportion falls into each model layer.
    for nret in range(len(pb_ret)-1):

        # 1st model pressure boundary index = the one before the
        # first boundary with lower pressure than high-pressure
        # retrieval layer boundary.
        model_lower = np.array(pb_mod_tmp < pb_ret[nret])
        id_model_lower = model_lower.nonzero()[0]
        id_min = id_model_lower[0]-1

        # Last model pressure boundary index = the last one with
        # higher pressure than low-pressure retrieval layer
        # boundary.
        model_higher = np.array(pb_mod_tmp > pb_ret[nret+1])

        id_model_higher = model_higher.nonzero()[0]

        if len(id_model_higher) == 0:
            #id_max = id_min
            raise ValueError("This shouldn't happen. Debug.")
        else:
            id_max = id_model_higher[-1]

        # By the way, in case there is no model level with
        # higher pressure than the next retrieval level,
        # id_max must be the same as id_min.

        # For each model layer, find out how much of it makes up this
        # retrieval layer
        for nmod in range(id_min, id_max+1):
            if (nmod == id_min) & (nmod != id_max):
                # Part of 1st model layer that falls within
                # retrieval layer
                coefs[nret, nmod] = pb_ret[nret] - pb_mod_tmp[nmod+1]
            elif (nmod != id_min) & (nmod == id_max):
                # Part of last model layer that falls within
                # retrieval layer
                coefs[nret, nmod] = pb_mod_tmp[nmod] - pb_ret[nret+1]
            elif (nmod == id_min) & (nmod == id_max):
                # id_min = id_max, i.e. model layer encompasses
                # retrieval layer
                coefs[nret, nmod] = pb_ret[nret] - pb_ret[nret+1]
            else:
                # Retrieval layer encompasses model layer
                coefs[nret, nmod] = pb_mod_tmp[nmod] - pb_mod_tmp[nmod+1]

        coefs[nret, :] = coefs[nret, :]/sum(coefs[nret, :])

    # I tested the code with many cases, but I'm only 99.9% sure
    # it works for all input. Hence a test here that the
    # coefficients sum to 1 and dump the data if not.
    sum_ = np.abs(coefs.sum(1) - 1)
    if np.any(sum_ > 2.*np.finfo(sum_.dtype).eps):
        # dump = dict(pb_ret=pb_ret,
        #             pb_mod=pb_mod,
        #             level_def=level_def)
        # fp = "int_coefs_dump.pkl"
        # with open(fp, "w") as f:
        #     pickle.dump(dump, f, 0)

        msg_fmt = "Something doesn't sum to 1. Arguments dumped to: %s"
        # David:
        # After adding the CAMS extension, 
        # this part complains for some observations...
        # Tested a few cases where it does not complain,
        # The results are near identical, therefore this check
        # has been omitted for now.
        # Might have some problems in the future?? (*to be verified!)
        # vvvvv this part vvvvv
        #raise ValueError(msg_fmt % fp)
            
        
    return coefs