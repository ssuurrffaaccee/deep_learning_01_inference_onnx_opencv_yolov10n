set(COMMNON_LIBS processing my_onnx my_util ${OpenCV_LIBS} onnxruntime::onnxruntime)

add_executable(multi_models multi_models.cpp)
target_link_libraries(multi_models 
                    #   mobile_net_v2
                    #   resnet50
                    #   retinaface
                    #   segmentation
                    #   yolovx
                       yolov10
                      ${COMMNON_LIBS})
install(TARGETS multi_models RUNTIME DESTINATION bin)
