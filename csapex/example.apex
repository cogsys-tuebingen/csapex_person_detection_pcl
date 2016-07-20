#!/localhome/kuhlmann/workspace/schoenbuch/devel/lib/csapex/csapex_node
threads:
  assignments:
    - id: 1
      uuid: ~
    - id: 1
      uuid: csapex::FileImporter_0
    - id: 1
      uuid: person_detection::PCLPersonDetector_0
    - id: 1
      uuid: csapex::LabelClusteredPointCloud_0
    - uuid: csapex::ColorPointCloud_0
      id: 1
    - id: 1
      uuid: csapex::TransformCloud_0
    - id: 1
      uuid: csapex::CloudRenderer_0
    - uuid: csapex::StaticTransform_1
      id: 1
    - id: 1
      uuid: csapex::PassThroughByCluster_0
    - uuid: csapex::CloudRenderer_3
      id: 1
    - id: 1
      uuid: person_detection::GroundPlaneEstimation_1
  groups: ~
nodes:
  - z: 501
    minimized: false
    type: csapex::FileImporter
    uuid: csapex::FileImporter_0
    label: csapex::FileImporter_0
    pos:
      - -636
      - 37
    color:
      - -1
      - -1
      - -1
    enabled: true
    flipped: false
    state:
      params:
        bag/play:
          type: value
          name: bag/play
          bool: true
        bag/publish clock:
          type: value
          name: bag/publish clock
          bool: false
        bag/frame:
          min: 0
          type: range
          name: bag/frame
          int: 139
          max: 176
          step: 1
        bag/loop:
          type: value
          name: bag/loop
          bool: true
        bag/latch:
          type: value
          name: bag/latch
          bool: true
        playback/frequency:
          type: range
          name: playback/frequency
          double: 5
          min: 1
          max: 256
          step: 0.5
        playback/immediate:
          type: value
          name: playback/immediate
          bool: false
        playback/resend:
          type: value
          name: playback/resend
          bool: false
        recursive import:
          type: value
          name: recursive import
          bool: false
        topic:
          type: set
          name: topic
          txt: /camera/depth_registered/points
          string: /camera/depth_registered/points
        directory/current_file:
          type: value
          name: directory/current_file
          string: ""
        directory/latch:
          type: value
          name: directory/latch
          bool: false
        directory/loop:
          type: value
          name: directory/loop
          bool: true
        directory/play:
          type: value
          name: directory/play
          bool: true
        directory/quit on end:
          type: value
          name: directory/quit on end
          bool: false
        directory/show parameters:
          type: value
          name: directory/show parameters
          bool: false
        directory/sort_numerically:
          type: value
          name: directory/sort_numerically
          bool: false
        import directory:
          type: value
          name: import directory
          bool: false
        path:
          type: path
          name: path
          value: /localhome/kuhlmann/data/icra/2016-08-01/2016-08-01-13-15-20.bag
        bag/publish tf:
          type: value
          name: bag/publish tf
          bool: false
        cache:
          type: value
          name: cache
          bool: false
        directory:
          type: path
          name: directory
          value: ""
        directory/current:
          type: range
          name: directory/current
          int: 0
          min: 0
          max: 1
          step: 1
      persistent_params:
        []
  - type: person_detection::PCLPersonDetector
    uuid: person_detection::PCLPersonDetector_0
    label: person_detection::PCLPersonDetector_0
    pos:
      - 209
      - -259
    color:
      - -1
      - -1
      - -1
    z: 498
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        classifier:
          type: path
          name: classifier
          value: /localhome/kuhlmann/workspace/schoenbuch/src/csapex/plugins/person_detection_pcl/config/trainedLinearSVMForPeopleDetectionWithHOG.yaml
        classifier_threshold:
          type: range
          name: classifier_threshold
          double: -999.9
          min: -1000
          max: 1000
          step: 0.01
        cluster_size:
          type: interval
          name: cluster_size
          int:
            - 0
            - 100000
          min: 0
          max: 100000
          step: 1
        fov_limit:
          type: interval
          name: fov_limit
          double:
            - 0
            - 100
          min: 0
          max: 100
          step: 0.01
        head_centroid:
          type: value
          name: head_centroid
          bool: true
        height_limit:
          type: interval
          name: height_limit
          double:
            - 1.3
            - 2.2
          min: 0
          max: 3
          step: 0.01
        instrinsic/c_x:
          type: value
          name: instrinsic/c_x
          int: 314
        instrinsic/c_y:
          type: value
          name: instrinsic/c_y
          int: 235
        instrinsic/fov_x:
          type: value
          name: instrinsic/fov_x
          double: 570
        instrinsic/fov_y:
          type: value
          name: instrinsic/fov_y
          double: 570
        minimum_head_distance:
          type: range
          name: minimum_head_distance
          double: 0.3
          min: 0
          max: 2
          step: 0.01
        sampling_factor:
          type: range
          name: sampling_factor
          int: 4
          min: 1
          max: 64
          step: 1
        sensor_potrait_orientation:
          type: value
          name: sensor_potrait_orientation
          bool: false
        voxel_size:
          type: range
          name: voxel_size
          double: 0.06
          min: 0
          max: 2
          step: 0.01
        width_limit:
          type: interval
          name: width_limit
          double:
            - 0.2
            - 8
          min: 0
          max: 10
          step: 0.01
      persistent_params:
        []
  - type: csapex::LabelClusteredPointCloud
    uuid: csapex::LabelClusteredPointCloud_0
    label: csapex::LabelClusteredPointCloud_0
    pos:
      - -55
      - 383
    color:
      - -1
      - -1
      - -1
    z: 478
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        {}
      persistent_params:
        []
  - uuid: csapex::ColorPointCloud_0
    color:
      - -1
      - -1
      - -1
    pos:
      - 88
      - 567
    enabled: true
    label: csapex::ColorPointCloud_0
    z: 234
    minimized: false
    flipped: false
    type: csapex::ColorPointCloud
    state:
      params:
        {}
      persistent_params:
        []
  - uuid: csapex::CloudRenderer_0
    type: csapex::CloudRenderer
    label: csapex::CloudRenderer_0
    pos:
      - 398
      - 349
    color:
      - -1
      - -1
      - -1
    z: 499
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        color/background:
          type: color
          name: color/background
          values:
            - 255
            - 255
            - 255
        color/field:
          type: set
          name: color/field
          txt: rgb
          string: rgb
        color/force gradient:
          type: value
          name: color/force gradient
          bool: false
        color/gradient/end:
          type: color
          name: color/gradient/end
          values:
            - 0
            - 0
            - 255
        color/gradient/start:
          type: color
          name: color/gradient/start
          values:
            - 0
            - 255
            - 0
        color/grid:
          type: color
          name: color/grid
          values:
            - 0
            - 0
            - 0
        color/rainbow:
          type: value
          name: color/rainbow
          bool: false
        point/size:
          type: range
          name: point/size
          double: 5
          min: 1
          max: 30
          step: 0.1
        show axes:
          type: value
          name: show axes
          bool: true
        ~grid/resolution:
          type: range
          name: ~grid/resolution
          double: 1
          min: 0.1
          max: 10
          step: 0.1
        ~grid/size:
          type: range
          name: ~grid/size
          int: 10
          min: 1
          max: 30
          step: 1
        ~grid/xy:
          type: value
          name: ~grid/xy
          bool: true
        ~grid/xz:
          type: value
          name: ~grid/xz
          bool: false
        ~grid/yz:
          type: value
          name: ~grid/yz
          bool: false
        ~size/height:
          type: range
          name: ~size/height
          int: 591
          min: 10
          max: 1024
          step: 1
        ~size/out/height:
          type: range
          name: ~size/out/height
          int: 591
          min: 10
          max: 1024
          step: 1
        ~size/out/sync:
          type: value
          name: ~size/out/sync
          bool: true
        ~size/out/width:
          type: range
          name: ~size/out/width
          int: 737
          min: 10
          max: 1024
          step: 1
        ~size/width:
          type: range
          name: ~size/width
          int: 737
          min: 10
          max: 1024
          step: 1
        ~view/dx:
          type: range
          name: ~view/dx
          double: 0
          min: -10
          max: 10
          step: 0.01
        ~view/dy:
          type: range
          name: ~view/dy
          double: 0
          min: -10
          max: 10
          step: 0.01
        ~view/dz:
          type: range
          name: ~view/dz
          double: 0
          min: -10
          max: 10
          step: 0.01
        ~view/phi:
          type: range
          name: ~view/phi
          double: -0.04000000000000102
          min: -3.141592653589793
          max: 3.141592653589793
          step: 0.001
        ~view/r:
          type: range
          name: ~view/r
          double: 8.799999999999994
          min: 0.01
          max: 20
          step: 0.01
        ~view/theta:
          type: range
          name: ~view/theta
          double: 1.740999999999999
          min: 0
          max: 3.141592653589793
          step: 0.001
      persistent_params:
        []
  - uuid: csapex::TransformCloud_0
    label: csapex::TransformCloud_0
    color:
      - -1
      - -1
      - -1
    pos:
      - 654
      - 205
    minimized: false
    state:
      params:
        {}
      persistent_params:
        []
    z: 479
    enabled: true
    flipped: false
    type: csapex::TransformCloud
  - pos:
      - -617
      - 525
    color:
      - -1
      - -1
      - -1
    type: csapex::StaticTransform
    uuid: csapex::StaticTransform_1
    label: csapex::StaticTransform_0
    z: 363
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        child_frame:
          type: value
          name: child_frame
          string: /camera_link
        dx:
          type: value
          name: dx
          double: 0
        dy:
          type: value
          name: dy
          double: 0
        dz:
          type: value
          name: dz
          double: 0.763
        frame:
          type: value
          name: frame
          string: /base_link
        pitch:
          type: angle
          name: pitch
          value: 0
          min: -3.141592653589793
          max: 3.141592653589793
        roll:
          type: angle
          name: roll
          value: -1.570796326794897
          min: -3.141592653589793
          max: 3.141592653589793
        yaw:
          type: angle
          name: yaw
          value: -1.570796326794897
          min: -3.141592653589793
          max: 3.141592653589793
      persistent_params:
        []
  - uuid: csapex::CloudRenderer_3
    label: csapex::CloudRenderer_3
    color:
      - -1
      - -1
      - -1
    pos:
      - 1185
      - -571
    z: 447
    type: csapex::CloudRenderer
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        color/background:
          type: color
          name: color/background
          values:
            - 255
            - 255
            - 255
        color/field:
          type: set
          name: color/field
          txt: rgb
          string: rgb
        color/force gradient:
          type: value
          name: color/force gradient
          bool: false
        color/gradient/end:
          type: color
          name: color/gradient/end
          values:
            - 0
            - 0
            - 255
        color/gradient/start:
          type: color
          name: color/gradient/start
          values:
            - 0
            - 255
            - 0
        color/grid:
          type: color
          name: color/grid
          values:
            - 0
            - 0
            - 0
        color/rainbow:
          type: value
          name: color/rainbow
          bool: false
        point/size:
          type: range
          name: point/size
          double: 5
          min: 1
          max: 30
          step: 0.1
        show axes:
          type: value
          name: show axes
          bool: true
        ~grid/resolution:
          type: range
          name: ~grid/resolution
          double: 1
          min: 0.1
          max: 10
          step: 0.1
        ~grid/size:
          type: range
          name: ~grid/size
          int: 10
          min: 1
          max: 30
          step: 1
        ~grid/xy:
          type: value
          name: ~grid/xy
          bool: true
        ~grid/xz:
          type: value
          name: ~grid/xz
          bool: false
        ~grid/yz:
          type: value
          name: ~grid/yz
          bool: false
        ~size/height:
          type: range
          name: ~size/height
          int: 1694
          min: 10
          max: 1024
          step: 1
        ~size/out/height:
          type: range
          name: ~size/out/height
          int: 1694
          min: 10
          max: 1024
          step: 1
        ~size/out/sync:
          type: value
          name: ~size/out/sync
          bool: true
        ~size/out/width:
          type: range
          name: ~size/out/width
          int: 822
          min: 10
          max: 1024
          step: 1
        ~size/width:
          type: range
          name: ~size/width
          int: 822
          min: 10
          max: 1024
          step: 1
        ~view/dx:
          type: range
          name: ~view/dx
          double: -0.954831600189209
          min: -10
          max: 10
          step: 0.01
        ~view/dy:
          type: range
          name: ~view/dy
          double: 0.03561419993638992
          min: -10
          max: 10
          step: 0.01
        ~view/dz:
          type: range
          name: ~view/dz
          double: 0
          min: -10
          max: 10
          step: 0.01
        ~view/phi:
          type: range
          name: ~view/phi
          double: -0.01999999999999983
          min: -3.141592653589793
          max: 3.141592653589793
          step: 0.001
        ~view/r:
          type: range
          name: ~view/r
          double: 17.20000000000001
          min: 0.01
          max: 20
          step: 0.01
        ~view/theta:
          type: range
          name: ~view/theta
          double: 1.710592653589794
          min: 0
          max: 3.141592653589793
          step: 0.001
      persistent_params:
        []
  - uuid: csapex::PassThroughByCluster_0
    label: csapex::PassThroughByCluster_0
    color:
      - -1
      - -1
      - -1
    pos:
      - 792
      - -42
    minimized: false
    z: 441
    enabled: true
    flipped: false
    state:
      params:
        keep organized:
          type: value
          name: keep organized
          bool: true
      persistent_params:
        []
    type: csapex::PassThroughByCluster
  - type: person_detection::GroundPlaneEstimation
    uuid: person_detection::GroundPlaneEstimation_1
    label: person_detection::GroundPlaneEstimation_1
    pos:
      - -287
      - -109
    color:
      - -1
      - -1
      - -1
    z: 500
    minimized: false
    enabled: true
    flipped: false
    state:
      params:
        height/value:
          type: range
          name: height/value
          double: 0.7000000000000001
          min: 0
          max: 10
          step: 0.01
        initialize:
          type: trigger
          name: initialize
        strategy:
          type: set
          name: strategy
          txt: height
          int: 1
      persistent_params:
        []
connections:
  - uuid: csapex::FileImporter_0:|:event_0
    targets:
      - person_detection::GroundPlaneEstimation_1:|:slot_0
    types:
      - default
  - types:
      - default
    targets:
      - csapex::TransformCloud_0:|:in_0
    uuid: person_detection::PCLPersonDetector_0:|:out_2
  - uuid: person_detection::GroundPlaneEstimation_1:|:out_0
    targets:
      - person_detection::PCLPersonDetector_0:|:in_1
    types:
      - default
  - uuid: csapex::PassThroughByCluster_0:|:out_0
    targets:
      - csapex::CloudRenderer_3:|:in_0
    types:
      - default
  - uuid: csapex::LabelClusteredPointCloud_0:|:out_0
    targets:
      - csapex::ColorPointCloud_0:|:in_0
    types:
      - default
  - uuid: csapex::ColorPointCloud_0:|:out_0
    targets:
      - csapex::CloudRenderer_0:|:in_0
    types:
      - default
  - uuid: csapex::StaticTransform_1:|:out_0
    targets:
      - csapex::TransformCloud_0:|:in_1
    types:
      - default
  - uuid: person_detection::PCLPersonDetector_0:|:out_0
    targets:
      - csapex::LabelClusteredPointCloud_0:|:in_1
      - csapex::PassThroughByCluster_0:|:in_1
    types:
      - default
      - default
  - uuid: csapex::TransformCloud_0:|:out_0
    targets:
      - csapex::LabelClusteredPointCloud_0:|:in_0
      - csapex::PassThroughByCluster_0:|:in_0
    types:
      - default
      - default
  - uuid: csapex::FileImporter_0:|:out_1
    targets:
      - person_detection::PCLPersonDetector_0:|:in_0
      - person_detection::GroundPlaneEstimation_1:|:in_0
    types:
      - default
      - default
adapters:
  []
uuid_map:
  csapex::CloudRenderer: 4
  csapex::ColorPointCloud: 1
  csapex::FileImporter: 1
  csapex::ImageToPointCloud: 1
  csapex::LabelClusteredPointCloud: 1
  csapex::ExportRos: 1
  person_detection::GroundPlaneEstimation: 2
  vision_plugins::ColorConvert: 1
  person_detection::PCLPersonDetector: 1
  csapex::RenderROIs: 1
  csapex::StaticTransform: 3
  csapex::TransformCloud: 1
  csapex::PassThroughByCluster: 1
  csapex::PointCloudToColorImage: 1
  csapex::PointCloudToIntensityImage: 1
  csapex::TransformPublisher: 1
  csapex::TextDisplay: 1