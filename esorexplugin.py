# This file is part of EsoReflex
# Copyright (C) 2017 European Southern Observatory
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

# Author: Artur Szostak <aszostak@partner.eso.org>

"""
This module implements a light weight layer that presents a more convenient API
to the developer writing Python based recipes. This abstracts away the very
minimalistic interface and protocol required by EsoRex to communicate with the
recipe. The exchange protocol is effectively just JSON text, sent back and forth
between EsoRex and the Python interpreter binaries over an input and output
Unix pipe.

When writing a new recipe plugin you should import this module in the following
manner:

  import esorexplugin
  from esorexplugin import *

This will make various helper functions and classes available, including the
esorexplugin.RecipePlugin class, which your own recipe must inherit. Note that
you must not import the esorexplugin.RecipePlugin class directly into the recipe
source code file, i.e. do not use "from esorexplugin import RecipePlugin". This
will cause EsoRex to mistakenly think that RecipePlugin is the implementation.
For more documentation details on the esorexplugin.RecipePlugin class, run the
following in your python shell:

  import esorexplugin
  help(esorexplugin.RecipePlugin)
"""

import os
import re
import json
import inspect
import traceback
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits


# Indicate the default classes to export if using 'from esorexplugin import *'.
# We avoid exporting CplPlugin or RecipePlugin because these will cause EsoRex
# to assume these are valid plugins when they are not.
__all__ = ['ValueParameter', 'RangeParameter', 'EnumParameter', 'Frame',
           'FrameData', 'VersionNumber', 'EsoCopyright']


class CplPlugin(object):
    """
    Minimal base class that must be inherited from for EsoRex to recognise the
    child classes as plugins implementing Python based recipes. EsoRex looks for
    all classes that derive from a class called CplPlugin.
    """

    def execute(self, data):
        raise NotImplementedError("Must be overloaded in child plugin class")


class RecipeError(Exception):
    """
    An exception class raised by the RecipePlugin.raise_error method.
    """
    def __init__(self, message, exitcode = 1, print_traceback = True):
        """
        Constructor that takes a message and desired return code (exitcode) that
        should be returned by the RecipePlugin.execute method. exitcode must be
        a non-zero integer.
        If print_traceback is set to True then the Python traceback will be
        printed before returning from RecipePlugin.execute().
        """
        self.exitcode = exitcode
        self.print_traceback = print_traceback
        Exception.__init__(self, message)


def VersionNumber(major, minor, micro):
    """
    Helper function to prepare a recipe version number for the 'version'
    attribute of a class deriving from RecipePlugin. This accepts 3 positive
    integer numbers that form part of a complete recipe version number, the
    major version for major backwards incompatible changes, minor version number
    for large but backwards compatible changes, and the micro version number for
    bug fixes that do not add or change features.

    Example usage:

       version = VersionNumber(1, 2, 3)

    This would assign a version number equivalent to the version string "1.2.3".

    Note that the minor and micro version numbers must be smaller than 100.
    """
    if not isinstance(major, int):
        raise TypeError("The major version number must be an integer type")
    if major < 0:
        raise ValueError("The major version number must be positive")
    if not isinstance(minor, int):
        raise TypeError("The minor version number must be an integer type")
    if minor < 0 or 99 < minor:
        msg = "The minor version number must be in the range [0..99]"
        raise ValueError(msg)
    if not isinstance(micro, int):
        raise TypeError("The micro version number must be an integer type")
    if micro < 0 or 99 < micro:
        msg = "The micro version number must be in the range [0..99]"
        raise ValueError(msg)
    return major * 10000 + minor * 100 + micro


def _is_string(obj):
    """
    Check if the object is a string and return True if it is else False if it is
    not. This function should work in Python 2 and 3.
    """
    if isinstance(obj, str):
        return True
    try:
        if isinstance(obj, basestring):
            return True
    except NameError:
        pass
    return False


def _create_param(name, class_name, default_value, idnum, description, context,
                  tag, cli_enabled, cli_alias, env_enabled, env_alias,
                  cfg_enabled, cfg_alias, displayName, min_value, max_value,
                  choices):
    """
    Constructs a dictionary representing a recipe parameter object that can be
    converted directly to JSON text by the 'json' module. This is a helper
    function to implement the ValueParameter, RangeParameter and EnumParameter
    functions.
    """

    if not _is_string(name):
        raise TypeError("The name parameter must be a string type")
    if not _is_string(description):
        raise TypeError("The description parameter must be a string type")
    if not _is_string(context):
        raise TypeError("The context parameter must be a string type")

    if class_name == 'range':
        if not (isinstance(default_value, int) \
                and not isinstance(default_value, bool)) \
           and not isinstance(default_value, float):
            msg = "The default_value parameter must be an integer or float type"
            raise TypeError(msg)
        if type(min_value) is not type(default_value):
            msg = "The min_value parameter has a different type than" \
                  " default_value"
            raise TypeError(msg)
        if type(max_value) is not type(default_value):
            msg = "The max_value parameter has a different type than" \
                  " default_value"
            raise TypeError(msg)
        if max_value < min_value:
            msg = "The max_value parameter must be bigger than min_value"
            raise ValueError(msg)
        if default_value < min_value or default_value > max_value:
            msg = "The default_value parameter must be in the range" \
                  " [min_value .. max_value]"
            raise ValueError(msg)
    elif class_name == 'enum':
        if not _is_string(default_value) \
           and not (isinstance(default_value, int) \
                    and not isinstance(default_value, bool)) \
           and not isinstance(default_value, float):
            msg = "The default_value parameter must be a string, integer or" \
                  " float type"
            raise TypeError(msg)
        if not isinstance(choices, list):
            raise TypeError("The choices parameter must be a list")
        if len(choices) < 1:
            raise ValueError("The choices list must contain at least one entry")
        if default_value not in choices:
            raise ValueError("The choices list must contain default_value")
        for entry in choices:
            if type(entry) is not type(default_value):
                msg = "Entries in the choices list have a different type than" \
                      " default_value"
                raise TypeError(msg)
    else:
        if not _is_string(default_value) \
           and not isinstance(default_value, int) \
           and not isinstance(default_value, float) \
           and not isinstance(default_value, bool):
            msg = "The default_value parameter must be a string, integer," \
                  " float or boolean type"
            raise TypeError(msg)

    param = {'name': name,
             'class': class_name,
             'description': description,
             'context': context,
             'default': default_value}

    if idnum is not None:
        if not isinstance(idnum, int):
            raise TypeError("The idnum parameter must be an integer type")
        param['id'] = idnum
    if tag is not None:
        if not _is_string(tag):
            raise TypeError("The tag parameter must be a string type")
        param['tag'] = tag
    cli_already_used = False
    if cli_enabled is not None:
        if not isinstance(cli_enabled, bool):
            raise TypeError("The cli_enabled parameter must be a boolean type")
        param['cli_enabled'] = cli_enabled
        cli_already_used = True
    if cli_alias is not None:
        if not _is_string(cli_alias):
            raise TypeError("The cli_alias parameter must be a string type")
        param['cli_alias'] = cli_alias
        cli_already_used = True
    if displayName is not None:
        if cli_already_used:
            msg = "Cannot use displayName and cli_alias or cli_enabled at the" \
                  " same time"
            raise SyntaxError(msg)
        if not _is_string(displayName):
            raise TypeError("The displayName parameter must be a string type")
        param['cli_alias'] = displayName
        param['cli_enabled'] = True
    if env_enabled is not None:
        if not isinstance(env_enabled, bool):
            raise TypeError("The env_enabled parameter must be a boolean type")
        param['env_enabled'] = env_enabled
    if env_alias is not None:
        if not _is_string(env_alias):
            raise TypeError("The env_alias parameter must be a string type")
        param['env_alias'] = env_alias
    if cfg_enabled is not None:
        if not isinstance(cfg_enabled, bool):
            raise TypeError("The cfg_enabled parameter must be a boolean type")
        param['cfg_enabled'] = cfg_enabled
    if cfg_alias is not None:
        if not _is_string(cfg_alias):
            raise TypeError("The cfg_alias parameter must be a string type")
        param['cfg_alias'] = cfg_alias

    if min_value is not None:
        param['min'] = min_value
    if max_value is not None:
        param['max'] = max_value
    if choices is not None:
        param['choices'] = choices

    return param


def ValueParameter(name, default_value, idnum = None,
                   description = "No description", context = "unknown",
                   tag = None, cli_enabled = None, cli_alias = None,
                   env_enabled = None, env_alias = None, cfg_enabled = None,
                   cfg_alias = None, displayName = None):
    """
    This helper function constructs a definition of a recipe parameter that can
    be either a boolean, integer, floating-point number or string. The type is
    derived from the type of the default_value.
    At a minimum, one must provide a name for the parameter and its default
    value. For example:

        ValueParameter('par1', 3)

    The name should be the same name as will be used in the argument list of the
    overridden RecipePlugin.process() function. The above example sets the
    default value for the parameter to 3.
    The following additional options can be specified for the recipe parameter:

        description - Text describing the parameters use in the recipe.
        context - The EsoRex context of this parameter, i.e. the specific
            component to which this belongs.
        tag - An optional tag string associated with the parameter.
        cli_enabled - Boolean indicating if the parameter should be configurable
            through the EsoRex command line interface.
        cli_alias - The parameter's alias to accept as a command line argument.
        env_enabled - Boolean indicating if the parameter should be configurable
            through an environment variable.
        env_alias - The parameter's alias to accept for environment variables.
        cfg_enabled - Boolean indicating if the parameter should be configurable
            through an EsoRex configuration file.
        cfg_alias - The parameter's alias to accept inside configuration files.
        displayName - This parameter follows the PythonActor naming convention
            and is equivalent to setting cli_alias and forcing cli_enabled=True.
        idnum - Forces the ID number associated with the parameter.

    Note that it is advisable to always add the description and context
    information.
    Currently EsoRex ignores the idnum, env_enabled and env_alias values. Thus,
    although they get forwarded to EsoRex, setting them has no effect at the
    moment.
    """
    return _create_param(name, 'value', default_value, idnum, description,
                         context, tag, cli_enabled, cli_alias, env_enabled,
                         env_alias, cfg_enabled, cfg_alias, displayName, None,
                         None, None)


def RangeParameter(name, default_value, min_value, max_value, idnum = None,
                   description = "No description", context = "unknown",
                   tag = None, cli_enabled = None, cli_alias = None,
                   env_enabled = None, env_alias = None, cfg_enabled = None,
                   cfg_alias = None, displayName = None):
    """
    This helper function constructs a definition of a recipe parameter that can
    be an integer or floating-point range. One must provide a name for the
    parameter, it's default value, and the minimum and maximum valid values that
    the parameter can accept. For example:

        RangeParameter('par1', 3, 1, 5)

    The name should be the same name as will be used in the argument list of the
    overridden RecipePlugin.process() function. The above example will create
    a range parameter that can accept any value x, where 1 <= x and x <= 5. The
    default value for the parameter is set to 3.
    The following additional options can be specified for the recipe parameter:

        description - Text describing the parameters use in the recipe.
        context - The EsoRex context of this parameter, i.e. the specific
            component to which this belongs.
        tag - An optional tag string associated with the parameter.
        cli_enabled - Boolean indicating if the parameter should be configurable
            through the EsoRex command line interface.
        cli_alias - The parameter's alias to accept as a command line argument.
        env_enabled - Boolean indicating if the parameter should be configurable
            through an environment variable.
        env_alias - The parameter's alias to accept for environment variables.
        cfg_enabled - Boolean indicating if the parameter should be configurable
            through an EsoRex configuration file.
        cfg_alias - The parameter's alias to accept inside configuration files.
        displayName - This parameter follows the PythonActor naming convention
            and is equivalent to setting cli_alias and forcing cli_enabled=True.
        idnum - Forces the ID number associated with the parameter.

    Note that it is advisable to always add the description and context
    information.
    Currently EsoRex ignores the idnum, env_enabled and env_alias values. Thus,
    although they get forwarded to EsoRex, setting them has no effect at the
    moment.
    """
    return _create_param(name, 'range', default_value, idnum, description,
                         context, tag, cli_enabled, cli_alias, env_enabled,
                         env_alias, cfg_enabled, cfg_alias, displayName,
                         min_value, max_value, None)


def EnumParameter(name, default_value, choices, idnum = None,
                  description = "No description", context = "unknown",
                  tag = None, cli_enabled = None, cli_alias = None,
                  env_enabled = None, env_alias = None, cfg_enabled = None,
                  cfg_alias = None, displayName = None):
    """
    This helper function constructs a definition of an enumeration type recipe
    parameter that can take selected values from a set of choices. The
    parameter's type can be an integer, floating-point or string. The type of
    the parameter is derived from the type of the default value.
    One must provide a name for the parameter, it's default value and a list of
    choices that the parameter can be set to. The default value must be one of
    the choices given. For example:

        EnumParameter('par1', 'A', ['A', 'B', 'C'])

    The name should be the same name as will be used in the argument list of the
    overridden RecipePlugin.process() function. The above example will create
    an enumeration parameter that can be set to one of the values 'A', 'B' or
    'C', and will be 'A' by default if not explicitly set by the user.
    The following additional options can be specified for the recipe parameter:

        description - Text describing the parameters use in the recipe.
        context - The EsoRex context of this parameter, i.e. the specific
            component to which this belongs.
        tag - An optional tag string associated with the parameter.
        cli_enabled - Boolean indicating if the parameter should be configurable
            through the EsoRex command line interface.
        cli_alias - The parameter's alias to accept as a command line argument.
        env_enabled - Boolean indicating if the parameter should be configurable
            through an environment variable.
        env_alias - The parameter's alias to accept for environment variables.
        cfg_enabled - Boolean indicating if the parameter should be configurable
            through an EsoRex configuration file.
        cfg_alias - The parameter's alias to accept inside configuration files.
        displayName - This parameter follows the PythonActor naming convention
            and is equivalent to setting cli_alias and forcing cli_enabled=True.
        idnum - Forces the ID number associated with the parameter.

    Note that it is advisable to always add the description and context
    information.
    Currently EsoRex ignores the idnum, env_enabled and env_alias values. Thus,
    although they get forwarded to EsoRex, setting them has no effect at the
    moment.
    """
    return _create_param(name, 'enum', default_value, idnum, description,
                         context, tag, cli_enabled, cli_alias, env_enabled,
                         env_alias, cfg_enabled, cfg_alias, displayName, None,
                         None, choices)


class RecipeParameter(object):
    """
    A simple recipe parameter object to which input JSON parameter information
    is converted before calling the RecipePlugin.process() method. The purpose
    of this object is to make accessing the parameter information more intuitive
    in the Python code, than would be the case if accessing dictionaries parsed
    directly from the JSON text.
    """

    def __init__(self, name, type, idnum, value, default, description,
                 context, tag, present, cli_enabled, cli_alias, env_enabled,
                 env_alias, cfg_enabled, cfg_alias, min_value = None,
                 max_value = None, choices = None):
        """
        Constructs an object corresponding to an input recipe parameter received
        from EsoRex. The following meta-data attributes are available:

        name - The name of the recipe parameter.
        type - Identifies the parameter type. It will be set to one of the
               following strings: 'value', 'range', 'enum'.
        idnum - An ID number assigned to the parameter by EsoRex.
        value - The value assigned to the parameter by EsoRex.
        default - The parameter's default value as was configured by the recipe.
        description - The parameter description as was configured by the recipe.
        context - The parameter's context as was configured by the recipe.
        tag - The parameter's tag string as was configured by the recipe.
        present - A boolean indcating if the parameters was set by EsoRex.
        cli_enabled - A boolean indicating if the parameter can be set on the
                      command line by EsoRex.
        cli_alias - The alias name of the parameter when set on the command
                    line.
        env_enabled - A boolean indicating if the parameter can be set with an
                      environment variable.
        env_alias - The name of the environment variable used to set the
                    parameter's value.
        cfg_enabled - A boolean indicating if the parameter can be set from a
                      configuration file.
        cfg_alias - The alias name of the parameter used in the configuration
                    file.
        min_value - This is the minimum valid value allowed for the parameter.
                    It is only valid if type == 'range'.
        max_value - This is the maximum valid value allowed for the parameter.
                    It is only valid if type == 'range'.
        choices - This is a list of allowed values for the parameter. It is only
                  valid if type == 'enum'.
        """
        self.name = name
        self.type = type
        self.idnum = idnum
        self.value = value
        self.default = default
        self.description = description
        self.context = context
        self.tag = tag
        self.present = present
        self.cli_enabled = cli_enabled
        self.cli_alias = cli_alias
        self.env_enabled = env_enabled
        self.env_alias = env_alias
        self.cfg_enabled = cfg_enabled
        self.cfg_alias = cfg_alias
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices

    def __repr__(self):
        return self.value.__repr__()


def _convert_parameters(parameters):
    """
    Converts a list of dictionaries into a list of RecipeParameter objects.
    The dictionaries must be from the parsed input JSON recipe parameter
    information that is received from EsoRex by this Python process.
    """

    def _get(param, name):
        return param[name] if name in param else None

    param_names = set()
    params_list = []
    for param in parameters:
        name = param['name']
        if name in param_names:
            msg = "Multiple parameters have been configured with the same" \
                  " name '{0}'".format(name)
            raise RecipeError(msg, exitcode = -1, print_traceback = True)
        param_names.add(name)
        par = RecipeParameter(name,
                    _get(param, 'class'),
                    _get(param, 'id'),
                    _get(param, 'value'),
                    _get(param, 'default'),
                    _get(param, 'description'),
                    _get(param, 'context'),
                    _get(param, 'tag'),
                    _get(param, 'present'),
                    _get(param, 'cli_enabled'),
                    _get(param, 'cli_alias'),
                    _get(param, 'env_enabled'),
                    _get(param, 'env_alias'),
                    _get(param, 'cfg_enabled'),
                    _get(param, 'cfg_alias'),
                    _get(param, 'min'),
                    _get(param, 'max'),
                    _get(param, 'choices')
                )
        params_list.append(par)
    return params_list


def _sort_parameters(parameters, config_params):
    """
    This function returns a sorted version of the parameters list (must be a
    list of RecipeParameter objects). The sorting is based on the order of the
    parameters in the config_params list, which must come from the recipe
    object's 'parameters' class attribute.
    """
    params = []
    lut = {}
    # Create a lookup table for the parameters list, from which we select the
    # correct parameter object according to the order in config_params.
    for param in parameters:
        lut[param.name] = param
    for param in config_params:
        name = param['name']
        params.append(lut[name])
    return params


class Frame(object):
    """
    A class representing a recipe input frame, typically a FITS file. The input
    JSON frame information will be converted to these objects before calling the
    RecipePlugin.process() method. This makes the manipulation of the frames
    more intuitive in the Python code, rather than manipulating the dictionaries
    that the JSON text is parsed into.
    """

    # The following are valid codes that can be used for the 'type' argument of
    # this classes constructor.
    TYPE_NONE     = 1 << 0
    TYPE_IMAGE    = 1 << 1
    TYPE_MATRIX   = 1 << 2
    TYPE_TABLE    = 1 << 3
    TYPE_PAF      = 1 << 4
    TYPE_ANY      = 1 << 5

    # Valid codes that can be used in the 'group' argument of the constructor.
    GROUP_NONE    = 0
    GROUP_RAW     = 1
    GROUP_CALIB   = 2
    GROUP_PRODUCT = 3

    # Valid codes that can be used in the 'level' argument of the constructor.
    LEVEL_NONE            = 0
    LEVEL_TEMPORARY       = 1
    LEVEL_INTERMEDIATE    = 2
    LEVEL_FINAL           = 3

    def __init__(self, filename, tag, type = None, group = None, level = None):
        """
        Creates a new recipe frame object having a corresponding file name and
        tag string, e.g. 'RAW', 'CALIB' etc. The following optional arguments
        can also be set:
          type - The type of the file indicated by filename. By default it is
            set to Frame.TYPE_ANY, but can be one of the following:
                Frame.TYPE_IMAGE, Frame.TYPE_MATRIX, Frame.TYPE_TABLE,
                Frame.TYPE_PAF, Frame.TYPE_ANY
          group - The overall category grouping information needed by the DFS
            system. Set to Frame.GROUP_PRODUCT by default, but can be one of the
            following:
                Frame.GROUP_RAW, Frame.GROUP_CALIB, Frame.GROUP_PRODUCT
          level - Indicates at which level of processing the product applies to,
            e.g. indicating if this is an intermediate product or a final one.
            By default this is set to Frame.LEVEL_FINAL, but can be one of:
                Frame.LEVEL_TEMPORARY, Frame.LEVEL_INTERMEDIATE,
                Frame.LEVEL_FINAL
        The reserved Frame.TYPE_NONE, Frame.GROUP_NONE and Frame.LEVEL_NONE
        values are also possible for their respective arguments, but should be
        used only in special cases.
        """
        if not _is_string(filename):
            raise TypeError("The filename parameter must be a string type")
        self.filename = filename

        if not _is_string(tag):
            raise TypeError("The tag parameter must be a string type")
        self.tag = tag

        if type is not None:
            if not isinstance(type, int):
                raise TypeError("The type parameter must be an integer type")
            self.type = type
        else:
            self.type = Frame.TYPE_ANY

        if group is not None:
            if not isinstance(group, int):
                raise TypeError("The group parameter must be an integer type")
            self.group = group
        else:
            self.group = Frame.GROUP_PRODUCT

        if level is not None:
            if not isinstance(level, int):
                raise TypeError("The level parameter must be an integer type")
            self.level = level
        else:
            self.level = Frame.LEVEL_FINAL

    def __repr__(self):
        return "Frame(filename = {0}, tag = {1}, type = {2}, group = {3}," \
               " level = {4})".format(self.filename.__repr__(),
                                      self.tag.__repr__(),
                                      self.type.__repr__(),
                                      self.group.__repr__(),
                                      self.level.__repr__())

    def open(self, **kwargs):
        """
        Opens the underlying FITS file and returns an hdulist object. Any
        additional option arguments given to this function are passed onto the
        astropy.io.open() function. See the Astropy documentation for allowed
        options.
        """
        return fits.open(self.filename, **kwargs)

    def write(self, hdulist, **kwargs):
        """
        Writes an hdulist object to a FITS file using the filename corresponding
        to this Frame object. Any additional option arguments are passed to the
        hdulist.writeto() method. See the Astropy documentation for allowed
        options and further details.
        Note that the overwrite = True option has been implemented by this
        method in such a way, that it will work even if an older underlying
        version of hdulist.writeto() does not support it.
        """
        try:
            hdulist.writeto(self.filename, **kwargs)
        except TypeError as error:
            errmsg = error.__str__()
            if "unexpected keyword argument 'overwrite'" in errmsg:
                # At this point try work around the missing support for the
                # overwrite option.
                if kwargs['overwrite'] and os.path.exists(self.filename):
                    os.remove(self.filename)
                del kwargs['overwrite']
                hdulist.writeto(self.filename, **kwargs)
            else:
                raise


def _convert_frames(frames):
    """
    Converts a list of dictionaries into Frame objects. The dictionaries are
    those from the JSON input text as parsed by the 'json' module. The JSON text
    is received by this Python process from EsoRex and contains the input frame
    information for the recipe.
    """
    frameset = []
    for frame in frames:
        frameset.append(Frame(frame['filename'], frame['tag'], frame['type'],
                              frame['group'], frame['level']))
    return frameset


def _frames_to_dictionaries(frames):
    """
    Converts a list of Frame objects back into a list of dictionaries that can
    be serialised into JSON text by the 'json' module.
    """
    frameset = []
    for frame in frames:
        frameset.append({'filename': frame.filename, 'tag': frame.tag,
                         'type': frame.type, 'group': frame.group,
                         'level': frame.level})
    return frameset


def FrameData(tag, min = 1, max = 1, inputs = None, outputs = None):
    """
    Helper function to construct an entry in the recipeconfig attribute of a
    class deriving from RecipePlugin. The recipeconfig data provides extra
    information about the frames that a recipe can handle or requires, and also
    indicates the relations between input and output frames.
    The only mandatory argument that must be provided is an appropriate frame
    tag. In addition, the minimum (with option min) and maximum (with option
    max) number of frames can be given. By default these are set to 1. It is
    possible to set min or max to None to mark the value as unspecified.
    Alternatively, use a value of 0 for min to indicate that the frame is
    optional. A list of associated input and output frames can also be given.
    The entries in the input frame list should be defined with this function as
    well. However only the tag, min and max function arguments should be used in
    that case. The outputs must be a list of frame tags of type string.
    """
    if not _is_string(tag):
        raise TypeError("The tag parameter must be a string type")
    if min is not None and not isinstance(min, int):
        raise TypeError("The min parameter must be an integer type")
    if min is not None and min < 0:
        raise ValueError("The min parameter must be a positive integer or None")
    if max is not None and not isinstance(max, int):
        raise TypeError("The max parameter must be an integer type")
    if max is not None and max < 0:
        raise ValueError("The max parameter must be a positive integer or None")
    if min is not None and max is not None and min > max:
        raise ValueError("The max parameter must greater than min")
    if min is None:
        min = -1
    if max is None:
        max = -1
    if inputs is not None:
        msg = "The inputs parameter must be a list of dictionaries"
        if not isinstance(inputs, list):
            raise TypeError(msg)
        for entry in inputs:
            if not isinstance(entry, dict):
                raise TypeError(msg)
    if outputs is not None:
        msg = "The outputs parameter must be a list of strings"
        if not isinstance(outputs, list):
            raise TypeError(msg)
        for entry in outputs:
            if not _is_string(entry):
                raise TypeError(msg)

    framedata = {'tag': tag, 'min': min, 'max': max}
    if inputs is not None:
        framedata['inputs'] = inputs
    if outputs is not None:
        framedata['outputs'] = outputs
    return framedata


class RecipePlugin(CplPlugin):
    """
    The base class for Python based EsoRex recipe plugins.
    New recipes must derive from this class and overload the set_frame_group()
    and process() methods.

    A number of class attributes should be declared in the child class deriving
    from this base class. These define the important parameters of the recipe
    such as its name, version and input parameters.
    The attributes to declare include the following:

        name - A string indicating the name of the
        version - The version number of the recipe encoded as an integer. The
            best way to construct this value is by using the VersionNumber()
            function.
        synopsis - A short description of the recipe.
        description - A string containing the full description of the recipe
            that will be used in when EsoRex generates the recipe's manpage.
        author - The author of the recipe.
        email - The email address where bug reports should be sent to.
        copyright - A copyright message string. One can either give a custom
            string message or use the EsoCopyright() function to produce a
            standard ESO GNU based copyright message.
        parameters - A list of dictionaries containing recipe input parameter
            descriptions. Each entry to the list should be created with one of
            the functions ValueParameter(), RangeParameter() or EnumParameter().
        recipeconfig - (optional) A list of dictionaries indicating frame types
            and their inputs and outputs. This allows to specify and forward the
            relationships between different types of frames to EsoRex. If this
            attribute is not given then the recipe defaults to a CPL version 1
            recipe plugin, otherwise if recipeconfig is given the recipe will be
            registered with EsoRex as a version 2 recipe.
    """

    def __init__(self):
        """
        Initialises default values for missing class attributes.
        """

        # Check if plugin configuration parameters are setup. If not, then set
        # some default values.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__
        if not hasattr(self, 'version'):
            self.version = 0
        try:
            if inspect.getdoc(self):
                have_doc_string = True
            else:
                have_doc_string = False
        except:
            have_doc_string = False
        if  not hasattr(self, 'synopsis') and not have_doc_string:
            self.synopsis = "no synopsis provided"
        if not hasattr(self, 'description') and not have_doc_string:
            self.description = "unknown"
        if not hasattr(self, 'author'):
            self.author = "unknown"
        if not hasattr(self, 'email'):
            self.email = "<unknown>"
        if not hasattr(self, 'copyright'):
            self.copyright = "unknown"
        if not hasattr(self, 'parameters'):
            self.parameters = []

    def execute(self, data):
        """
        The recipe entry point as called by EsoRex's Python interface. The
        'data' dictionary will contain the following key entries:

          'parameters' - The list of input parameters for the recipe as parsed
                from the JSON send by the EsoRex binary. This will be a list of
                dictionaries that will be converted to RecipeParameter objects
                and stored in self.input_parameters, so the self.process method
                can have access to the full parameter information later.
          'frames' - The list of parsed JSON input frame information. This will
                be a list of dictionaries that will be converted to Frame
                objects. This list will also be updated with the new output
                frames produced by the recipe's Python code.
          'recipeconfig' - (optional) the parsed JSON recipe configuration as
                was originally declared in the 'recipeconfig' class attribute by
                the child recipe class.

        This method must return an integer return code where 0 indicates success
        and a non-zero value indicates and error. If an error does occur then
        the self.error_message string attribute should also be set with an
        appropriate human readable message indicating the reason for the error.
        """

        try:
            # Check that parameters and frames are available in the data.
            if 'parameters' not in data:
                self.raise_error("The 'parameters' entry is missing from the"
                                 " received input data.",
                                 exitcode = -1, print_traceback = True)
            if 'frames' not in data:
                self.raise_error("The 'frames' entry is missing from the"
                                 " received input data.",
                                 exitcode = -1, print_traceback = True)

            # Convert raw Python structures parsed from the JSON input to
            # Frame and RecipeParameter objects. The parameter values are
            # additionally sorted according to the parameter configuration list
            # originally prepared in the 'parameters' class attribute.
            parameters = _convert_parameters(data['parameters'])
            parameters = _sort_parameters(parameters, self.parameters)
            frames = _convert_frames(data['frames'])

            # Have the recipe set the groups for the input frames.
            for frame in frames:
                self.set_frame_group(frame)

            # Set the input parameters to allow the child recipe access to them
            # if it needs it.
            self.input_parameters = parameters

            # Let the child recipe class process the frames. Pass the input
            # parameter values on as arguments to the process method. This makes
            # the method's code in the child recipe class more natural.
            output_frames = self.process(frames, *[p.value for p in parameters])
            if not isinstance(output_frames, list):
                self.raise_error("The process() method must return a list, but"
                                 " it returned {0} instead.".format(
                                     type(output_frames)),
                                 exitcode = -1, print_traceback = True)

            # Convert all the frames (including the output frames) back to
            # dictionaries for JSON encoding.
            data['frames'] = _frames_to_dictionaries(frames + output_frames)

            return 0

        except RecipeError as error:
            # Print the traceback if requested, set the error message and return
            # the set exit code.
            if error.print_traceback:
                traceback.print_exc()
            self.error_message = str(error)
            return error.exitcode

    def raise_error(self, message, exitcode = 1, print_traceback = False):
        """
        Raises a recipe exception to be handled by the EsoRex-Python interface.
        One must give an appropriate human readable message describing the
        reason for the recipe failure.
        Optionally one can set and return integer code with exitcode that will
        be returned by the recipe to EsoRex. This value must not be zero, since
        that indicates the recipe succeeded, it must be a non-zero integer.
        print_traceback can be set to True to additionally print a Python stack
        trace on standard error, just before the execute() entry point method
        returns.
        """
        raise RecipeError(message, exitcode, print_traceback)

    def set_frame_group(self, frame):
        """
        This method must be overloaded by the child to categorise the given
        frame, which will be a Frame object. It will be called for every input
        frame received by the recipe. The method should look at the frame's
        attributes such as the tag and decide which group and level the frame
        belongs to. The group and level fields must then be updated for the
        frame.
        This method should not return any value, only frame should be updated.
        """
        raise NotImplementedError("Must be overloaded in child recipe class")

    def process(self, frames, *args):
        """
        This method must be overloaded by the child class to implement the
        actual processing of the input frames. The input frames are given as a
        list of Frame objects in the frames argument. When new output frames are
        created they should be appended to a list of Frame objects and the whole
        list must be returned by this method once processing is complete.

        For every recipe parameter defined in the child recipe class, there must
        be a corresponding argument in the overloaded process() method. This is
        to allow passing of the input parameters to the recipe's processing
        stage. The names of the arguments in the process() method can be
        anything, but they will be filled in the same order as declared in the
        'parameters' class attribute list. For example, if 3 parameters are
        defined as follows in a recipe class (e.g. using ValueParameter), then
        the process() method should be declared with 3 additional positional
        attributes as follows:

            class Recipe(RecipePlugin):

                parameters = [
                        ValueParameter('par1', 1),
                        ValueParameter('par2', 2),
                        ValueParameter('test.par3', 3)
                    ]

                def process(self, frames, par1, par2, par3):
                    output_frames = []
                    ...
                    return output_frames

        Any errors occurring during processing should be indicated by calling
        the method raise_error() to raise an exception.
        """
        raise NotImplementedError("Must be overloaded in child recipe class")


def EsoCopyright(package_name, year):
    """
    Returns a string corresponding to an ESO copyright message based on GNU v2.
    package_name must be a string indicating the full name of the software
    package that the recipe belongs to. year can either be a string or integer
    indicating a specific year. Alternatively, year can be a list of integers
    each indicating a specific year. In that case, the years will be represented
    using appropriate ranges where possible in the resultant copyright message.
    """
    if _is_string(year):
        year_string = year
    elif isinstance(year, int):
        year_string = "{0}".format(year)
    else:
        # If the year argument is a list then try an format the years
        # appropriately, e.g. compress them into ranges where possible.
        ranges = []
        first_year = None
        previous_year = None
        for current_year in sorted(year):
            if first_year is None:
                first_year = current_year
            elif current_year - previous_year > 1:
                if first_year == previous_year:
                    ranges.append("{0}".format(first_year))
                else:
                    ranges.append("{0}-{1}".format(first_year, previous_year))
                first_year = current_year   # Start a new range.
            # Update previous year for next loop iteration:
            previous_year = current_year
        if first_year is not None:
            if first_year == previous_year:
                ranges.append("{0}".format(first_year))
            else:
                ranges.append("{0}-{1}".format(first_year, current_year))
        year_string = ", ".join(ranges)

    return \
    "This file is part of {0}\n"                                               \
    "Copyright (C) {1} European Southern Observatory\n"                        \
    "\n"                                                                       \
    "This program is free software; you can redistribute it and/or modify\n"   \
    "it under the terms of the GNU General Public License as published by\n"   \
    "the Free Software Foundation; either version 2 of the License, or\n"      \
    "(at your option) any later version.\n"                                    \
    "\n"                                                                       \
    "This program is distributed in the hope that it will be useful,\n"        \
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"         \
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"          \
    "GNU General Public License for more details.\n"                           \
    "\n"                                                                       \
    "You should have received a copy of the GNU General Public License\n"      \
    "along with this program; if not, write to the Free Software\n"            \
    "Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, \n"                \
    "MA  02111-1307  USA".format(package_name, year_string)
