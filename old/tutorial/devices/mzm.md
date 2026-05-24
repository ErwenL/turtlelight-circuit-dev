# Mach Zehnder Modulator (MZM)

## MzmDesign (siluxApi.models.devices.mzm.MzmDesign)

[MzmDesign](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign) class is a wrap-up of the MZM design methodolgy. It integrates the essential functions for designing a MZM, including:

- Querying TWE and PN junction data from database
- Calculating MZM frequency response
- Analyzing the MZM performance
- Parameter sweeping
- Parameter optimization

!!!warning "Backwards Compatibility"

    The `MzmDesign` data also serves as a documentation recording for our MZM design setting. Therefore, we emphasis on the **backwards compatibility**. While we keep integrating our latest understanding of MZM design into the `MzmDesign` class, we need to make sure that we can get consistent result from old `MzmDesign` data.


### Design Variables (`x`)

We use the key word `x` to represent the design variables, inspired by the `scipy.optimize.minimize`'s target function signature `fun(x, *args) -> float`. 

[`x`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.x) attribute is dictionary that contains the design variables.

- keys of `x` should match the `mzmdata.info["design_params"]` field, which together with `freqeuencies` define the param space of the mzm design been considered.
- design variables can be params of TWE, PN junction, `MzmDesign` itself.

[`x_mode`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.x_mode) attribute defines how to construct the param space from the `x` dict. It can be `meshgrid`, `array` and `point`.

[`x_bounds`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.x_bounds) attribute defines the bounds of the design variables when performing optimization.

[`set_x(**kwargs)`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.set_x) method is the setter method for the `x` attribute. Of course, you can directly modify the `x` dict. `set_x` does some additional checks for the input values.

[`load_x_2_data()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.load_x_2_data) method constructs the param space from the `x` dict and set it to the `data` attribute.

- Typically, `load_x_2_data()` is the first step to performance MZM calculation.
- You can also use another alias wrapper `init_x_param_space()` which is more intuitive.

!!!Notes "Param Optimization related `x` methods"

    When doing parameter optimization, it is very often that we started with a large param space, and then narrowed down to a smaller space by fix one or more design variables. Or reversely, we may start with a small param space that is easy to optimize, and then gradually adding more degree of freedom to the param space. The following methods are designed to facilitate such process:

    - [`fix_x`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.fix_x)
    - [`unfix_x`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.unfix_x)

    By Default, the `delayline_length` of a MZM is calculated for all the point in the param space based on velocity matching. However, in some case we may want to fix the `delayline_length` to a certain value, such as checking MZM performance versus bias for the final design, or reoptimizing MZM design for respin. The following methods are designed to facilitate such process:

    - [`fix_delayline_length`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.fix_delayline_length)
    - [`unfix_delayline_length`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.unfix_delayline_length)

    The target function `scipy.optimize.minimize` has the signature `fun(x, *args) -> float`, where `x` is 1D array. The following methods are designed to convert the optimization result (1D array) to `x` dict and vice versa:
    
    - [`update_x`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.update_x)
    - [`update_xs`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.update_xs)


!!!Notes "Param Sweeping related `x` methods"

    - `_MAX_SAMPLE_NUM = 50_000`
    - `rand_x_sample`
    - `meshgrid_x_sample`

### Segmented MZM

When designing the `MzmDesign` class, we consider a MZM as series of TWE segments. Each segment can be `active` (*PN junction loaded TWE*) or `passive` (*unload TWE and optical delayline*). Then, we can calculate the MZM frequency response given the properties of each TWE segment and the terminal resistance for the most simple MZM model. 

!!!Notes "MZM design parameters"

    The design parameters of a MZM can be divided into three categories:

    - TWE design params
        - such as: signal width, signal gap, etc.
    - PN junction design params
        - such as: junction offset, doping level, bias, etc
    - MZM design params
        - such as: terminal resistance, segment length, delayline length, etc

#### Define segments of a MZM

When calculating the properties of TWE segments, we first need to query the unload TWE and PN junction properties from the `MzmDesign.datasets`. The TWE or PN junction design params may be claimed in either `x` or `meta` attributes, we need to tell the program where to look for the params when performing data query. Such information is stored in the `segment_infos` attribute.

[`segment_infos`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.segment_infos) is a list of [`SegmentInfo`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.SegmentInfo) object.

!!!Note "types of `SegmentInfo`"

    - [`PairSegmentInfo`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.PairSegmentInfo)
        - A `PairSegmentInfo` is a pair of `ActiveSegmentInfo` and `PassiveSegmentInfo` that share the same design params.
        - When converting a `PairSegmentInfo` to segment datas, the `optical_length` of the passive segment is calculated based velocity matching, if `delayline_length` is not specified.
    - [`ActiveSegmentInfo`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.ActiveSegmentInfo)
    - [`PassiveSegmentInfo`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.PassiveSegmentInfo)
    - [`CircuitSegmentInfo`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.CircuitSegmentInfo)

[`get_seg_param`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.get_seg_param) method is used to lookup the segment parameter from the `MzmDesign` attributes based on the [`ParamMap`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.ParamMap) defined in the `SegmentInfo`


It is recommended to use the [`SegmentInfoHelper`](../../reference/models/devices/mzm/utils.md#models.devices.mzm.utils.segment.SegmentInfoHelper) to create the `segment_infos` list.

```py title="example of creating segment_infos"
from siluxApi.models.devices.mzm.utils.segment import SegmentInfoHelper

segment_infos = [
    SegmentInfoHelper.pair(
        param_map = {
            "signal_width": ("x", "signal_width"),
            "signal_gap": ("x", "signal_gap"),
            "bias": ("x", "bias"),
            "junction_offset": ("meta", "junction_offset")
        },
        active_seg_length = ("data", "active_seg_length"),
        passive_seg_length = ("meta", "passive_seg_length"),
        differential_twe = True
    ),
    SegmentInfoHelper.circuit(
        circuit: "terminal",
        param_map = {
            "r": ("meta", "terminal_resistance")
        }
    )
]
```

#### Uniform segments and nonuniform segments

By default, the MZM is typically conceptualized as a series of uniform segments alternating between active and passive TWE segments, in which case we just need define one `PairSegmentInfo` in the `segment_infos` and claim the number of (active) segments in the `meta["seg_num"]` field.

!!!warning "attention"

    Uniform segments defined by `PairSegmentInfo` always end with an active segment. The last passive segment will be ignored.

We can also consider the case of nonuniform segments, in which we can assign parameter to each TWE segment separately. We can do this by defining multiple `SegmentInfo` in the `segment_infos` list.

- You need to make sure that the total number of `PairSegmentInfo` and `ActiveSegmentInfo` in the `segment_infos` list match the `meta["seg_num"]` field.
- The lase TWE segment info in the `segment_infos` list should an `ActiveSegmentInfo`.

#### Construct segment datas

[`get_segment_datas`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.get_segment_datas) method constructs the segment datas based on the `segment_infos`. The results in the form as `pd.DataFrame` will be stored in `segment_datas` attribute, which will be used in MZM frequency response calculation.

#### Query datasets

When constructing TWE segment data, we need to query the TWE and PN junction datasets based on the segment's `ParamMap`.

[`query_seg_data(param_map:ParamMap)`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.query_seg_data) method first splits the `param_map` into three parts: `twe_params`, `pn_params` and `mzm_params`. Then, it queries the TWE and PN junction datasets based on the `twe_params` and `pn_params` respectively. Finally, it constructs the MZM design param space.

### Calculate MZM

#### Calculation Steps

[`init_datasets()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.init_datasets) method populates the `datasets` attributes with the TWE and PN junction datasets found from the database. Then, it initializes and processes the datasets based on fields in `info["datasets"]`.

`init_datasets()` should always be the first step after getting the `MzmDesign` from the database.

[`init_x_param_space()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.init_x_param_space) method constructs the param space from the `x` dict and set it to the `data` attribute.

[`get_segment_datas()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.get_segment_datas) method constructs the segment datas based on the `segment_infos`. The results in the form as `pd.DataFrame` will be stored in `segment_datas` attribute.

[`calculate_mzms()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.calculate_mzms) method calculates the MZM frequency response based on the segment datas and update the `data` attribute.

[`analyze_mzms()`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.analyze_mzms) method analyzes the MZM performance based on the MZM frequency response and store the results in the `summary` attribute. Analysis includes:

- `extrapolate_2_dc()`
- `calculate_sparams()`
- `calculate_bandwidth()`
- `calculate_extinction_ratio()`
- `calculate_loss()`
- `calculate_figure_of_merits()`
- calculate sparams in "dB" scale
- extract delayline lengths from `segment_datas`

[`update_mzm_data(complete:bool)`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesign.update_mzm_data) method is a wrapper of the above three methods.

- if `complete` is `True`, it will set `MzmDesign.config.complete_analysis` to `True` and perform the complete analysis.


```py title="example of calculating MZM"

design: MzmDesign
design.init_datasets()
design.init_x_param_space()
design.update_mzm_data(True)
```

#### MZM Design Config

`MzmDesign.config` attribute is a [`MzmDesignConfig`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig) object that stores the MZM design settings that are used during the calculation and analysis. It contains the following fields:


- [`terminal_as_circuit`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.terminal_as_circuit): whether the terminal is defined as a circuit.
- [`pad_as_circuit`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.pad_as_circuit): whether the RF pad is defined as a circuit.
- [`use_multiprocess`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.use_multiprocess): whether to perform the calculation in parallel. Setting it to an `int` number will enable multiprocess, the value is the number of processes to use.
- [`extrapolate_2_dc`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.extrapolate_2_dc): whether to extrapolate the MZM frequency response to DC. Default is `True`.
- [`complete_analysis`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.complete_analysis): whether to perform the complete analysis. Default is `False`.
- [`sampling_analysis`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.sampling_analysis): whether to perform sampling analysis. Default is `False`.
- [`use_linear_compensation`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.use_linear_compensation): whether to add linear compensation to calculated `phase_shift` frequency response during sampling analysis. Default is `False`.
- [`use_equalizer`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.use_equalizer): whether to use zero-forcing equalizer to compensate the MZM frequency response during sampling analysis. Default is `False`.
- [`er_method`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.er_method): domain to calculate the extinction ratio. Options: `t`, `f`.
- [`fom_method`](../../reference/models/devices/mzm/index.md#models.devices.mzm.design.MzmDesignConfig.fom_method): value name to use as the figure of merit.
    - Options: `gbp` (default), `inner_eye_height`, `noise_margin`

### Define Circuit Segment

`MzmDesign` supports define a `circuit.Composite` element as a segment of the MZM. The `circuit.Composite` element can be defined in the `MzmDesign.circuits` attribute. The `segment_infos` attribute should include a `CircuitSegmentInfo` object that defines the param map between the `circuit.Composite` element and the MZM design params.

#### Terminal and Pad as Circuit

By default, we assume `terminal_resistance` and `pad_capacitance`(optional) are provided in either `meta` or `x` attributes. When calculating the MZM frequency response, we will first look up these two values and then perform the calculation.

However, we can also define the terminal and pad as circuit elements. The steps are as follows:

- Define the `circuit.Composite` element for the terminal and pad, respectively.
- Add the terminal and pad elements to the `MzmDesign.circuits` attribute, which is a dictionary with the element name as the key and the `circuit.Composite` element as the value.
- Set the `terminal_as_circuit` and `pad_as_circuit` fields in the `MzmDesign.config` attribute to `True`, which will disable the default lookup of terminal resistance and pad capacitance.
    - If `terminal_as_circuit` is `True`, the last info of `segment_infos` attribute must be `CircuitSegmentInfo`, and the corresponding `circuit.Composite` element should be a 1-port element.
- Add the `CircuitSegmentInfo` object for the terminal and pad to the `segment_infos` attribute, which defines the param map between the circuit element and the MZM design params.
    - The `ParamMap` should cover all the params of the corresponding circuit element.


```py title="example of circuit segment"

# define composite elements
terminal = Composite(
    name="terminal",
    num_of_ports = 1,
    params = {
        "r": 80.,
    },
    params_listeners={
        "r": "Rterm::r"
    },
    circuit = Circuit(
        elements=[
            Resistor.new("Rterm", 80.)
        ],
        links=[
            "Rterm gnd p0"
        ]
    ),
    circuit_ports = ["p0"],
    z0=[100.]
)
pad = Composite(
    name="pad",
    num_of_ports = 2,
    params = {
        "c": 10e-15,
    },
    params_listeners={
        "c": "Cpad::c"
    },
    circuit = Circuit(
        elements=[
            Capacitor.new("Cpad", 10e-15)
        ],
        links = [
            "Cpad gnd p0"
        ]
    ),
    circuit_ports = ["p0", "p0"],
    z0=[100, 100]
)
terminal.connect()
pad.connect()

design:MzmDesign

# set `circuits` attribute
design.circuits = {
    "terminal": terminal,
    "pad": pad
}

# update `config`
design.config.terminal_as_circuit = True
design.config.pad_as_circuit = True

# update `segment_infos`
design.segment_infos = [
    # pad segment
    SegmentInfoHelper.circuit(
        circuit="pad",
        param_map = {
            "c": ("meta", "pad_capacitance")
        }
    ),
    # previous segment info
    design.segment_infos[0],
    # terminal segment
    SegmentInfoHelper.circuit(
        circuit="terminal",
        param_map = {
            "r": ("meta", "terminal_resistance")
        }
    )
]

```

### Sampling Analysis

`MzmDesign` supports evaluate the MZM performance based on impulse response in time domain, which can be enabled by modifying `MzmDesign.config`. Two additional columns `signal` and `interference` will be added to the `summary` dataframe.

```py title="config for sampling analysis"

design: MzmDesign

# enable sampling analysis
design.config.sampling_analysis = True

# enable linear compensation and equalizer
design.config.use_linear_compensation = True
design.config.use_equalizer = True

# calculate the extinction ratio in time domain
design.config.er_method = "t"

# set the figure of merit
design.config.fom_method = "inner_eye_height"
```

You can assign the parameters for sampling analysis in `meta` attribute, if you want to override the default values.
The following parameters are the default parameter values.

```py title="default sampling analysis parameters"
design: MzmDesign

# related to sampling analysis
design.meta["baud_rate"] = 53e9
design.meta["time_step"] = 1e-12
design.meta["symbol_length"] = 21

# related to linear compensation
design.meta["peaking_frequency"] = 40e9
design.meta["dc_gain"] = 0.
design.meta["peaking_gain"] = 0.
design.meta["roll_off"] = 20

# related to equalizer
design.meta["num_of_pre_taps"] = 2
design.meta["num_of_post_taps"] = 4
```

Except `config` and `meta`, you may also need to revise the `info["fom"]` field, because the FOM for sampling analysis requires different thresholds setting compared with `gbp`.

```py title="update FOM info"
from siluxApi.utils.fom import ScoreThreshold, Threshold

design: MzmDesign

design.info["fom"] = ScoreThreshold(
    # max or min, default is max, which can be omitted
    mode = "max", 
    # score name should match the FOM method
    score = "inner_eye_height", 
    # thresholds for the FOM, which is not required
    thresholds = {
        "loss": Threshold(
            target = 3,
            mode = "lt,
            power = 1
        )
    }
).dict() # convert `ScoreThreshold` to dict
```

```py title="get impulse analysis result"
design: MzmDesign

res = design.perform_impulse_analysis()

```



### Optimization 

```py title="example of optimization"
design: MzmDesign
log = design.optimize_fom(
    method = "Nelder-Mead",
    maxiter = 1000,
)
```
