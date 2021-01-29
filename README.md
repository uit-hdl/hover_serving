# This is repository for inference only code for HoVer-Net

Main repository is [hovernet-pipeline](https://github.com/uit-hdl/hovernet-pipeline)

All credits go to original authors 

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
