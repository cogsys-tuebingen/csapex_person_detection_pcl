#pragma once

/// PROJECT
#include <csapex/model/node.h>
#include "../../include/person_detection_pcl/hog.h"

/// EXTRACT HOG FEATURE

namespace person_detection
{

class PCLHOGExtractor : public csapex::Node
{
public:
    PCLHOGExtractor();

    void setupParameters(Parameterizable& parameters) override;
    void setup(csapex::NodeModifier& node_modifier) override;
    void process() override;

private:
    enum AdaptionType {SCALE, TRY_GROW, GROW_STRICT};

    pcl_backport::people::HOG hog_;

    csapex::Input  *in_img_;
    csapex::Input  *in_rois_;
    csapex::Output *out_;

    bool            mirror_;

    // theoretical parameters, since the detector doesn't allow changes, not included
    /*
    int             window_width_;
    int             window_height_;
    int             bin_size_;
    int             n_orients_;
    bool            soft_bin_;
    */
};

}
