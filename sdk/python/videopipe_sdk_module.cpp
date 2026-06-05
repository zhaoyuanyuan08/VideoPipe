#include <Python.h>

#include "../cpp/vp_pose_pipeline.h"

#include <memory>
#include <optional>
#include <string>

namespace {

    typedef struct {
        PyObject_HEAD
        vp_sdk::vp_pose_pipeline* pipeline;
    } PipelineObject;

    int set_item(PyObject* dict, const char* key, PyObject* value) {
        if (value == nullptr) {
            return -1;
        }
        auto rc = PyDict_SetItemString(dict, key, value);
        Py_DECREF(value);
        return rc;
    }

    PyObject* result_to_dict(const vp_sdk::vp_pipeline_result& result) {
        auto* dict = PyDict_New();
        if (dict == nullptr) {
            return nullptr;
        }
        if (set_item(dict, "channel_index", PyLong_FromLong(result.channel_index)) < 0 ||
            set_item(dict, "frame_index", PyLong_FromLong(result.frame_index)) < 0 ||
            set_item(dict, "width", PyLong_FromLong(result.width)) < 0 ||
            set_item(dict, "height", PyLong_FromLong(result.height)) < 0 ||
            set_item(dict, "fps", PyLong_FromLong(result.fps)) < 0 ||
            set_item(dict, "latency_ms", PyLong_FromLongLong(result.latency_ms)) < 0) {
            Py_DECREF(dict);
            return nullptr;
        }

        auto* targets = PyList_New(static_cast<Py_ssize_t>(result.targets.size()));
        if (targets == nullptr) {
            Py_DECREF(dict);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(result.targets.size()); ++i) {
            const auto& target = result.targets[static_cast<std::size_t>(i)];
            auto* item = PyDict_New();
            if (item == nullptr ||
                set_item(item, "x", PyLong_FromLong(target.x)) < 0 ||
                set_item(item, "y", PyLong_FromLong(target.y)) < 0 ||
                set_item(item, "width", PyLong_FromLong(target.width)) < 0 ||
                set_item(item, "height", PyLong_FromLong(target.height)) < 0 ||
                set_item(item, "class_id", PyLong_FromLong(target.class_id)) < 0 ||
                set_item(item, "score", PyFloat_FromDouble(target.score)) < 0 ||
                set_item(item, "label", PyUnicode_FromString(target.label.c_str())) < 0 ||
                set_item(item, "track_id", PyLong_FromLong(target.track_id)) < 0) {
                Py_XDECREF(item);
                Py_DECREF(targets);
                Py_DECREF(dict);
                return nullptr;
            }
            PyList_SET_ITEM(targets, i, item);
        }
        if (PyDict_SetItemString(dict, "targets", targets) < 0) {
            Py_DECREF(targets);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(targets);

        auto* poses = PyList_New(static_cast<Py_ssize_t>(result.poses.size()));
        if (poses == nullptr) {
            Py_DECREF(dict);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(result.poses.size()); ++i) {
            const auto& pose = result.poses[static_cast<std::size_t>(i)];
            auto* item = PyDict_New();
            auto* keypoints = PyList_New(static_cast<Py_ssize_t>(pose.keypoints.size()));
            if (item == nullptr || keypoints == nullptr || set_item(item, "type", PyLong_FromLong(pose.type)) < 0) {
                Py_XDECREF(item);
                Py_XDECREF(keypoints);
                Py_DECREF(poses);
                Py_DECREF(dict);
                return nullptr;
            }
            for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(pose.keypoints.size()); ++j) {
                const auto& keypoint = pose.keypoints[static_cast<std::size_t>(j)];
                auto* point = PyDict_New();
                if (point == nullptr ||
                    set_item(point, "type", PyLong_FromLong(keypoint.type)) < 0 ||
                    set_item(point, "x", PyLong_FromLong(keypoint.x)) < 0 ||
                    set_item(point, "y", PyLong_FromLong(keypoint.y)) < 0 ||
                    set_item(point, "score", PyFloat_FromDouble(keypoint.score)) < 0) {
                    Py_XDECREF(point);
                    Py_DECREF(keypoints);
                    Py_DECREF(item);
                    Py_DECREF(poses);
                    Py_DECREF(dict);
                    return nullptr;
                }
                PyList_SET_ITEM(keypoints, j, point);
            }
            if (PyDict_SetItemString(item, "keypoints", keypoints) < 0) {
                Py_DECREF(keypoints);
                Py_DECREF(item);
                Py_DECREF(poses);
                Py_DECREF(dict);
                return nullptr;
            }
            Py_DECREF(keypoints);
            PyList_SET_ITEM(poses, i, item);
        }
        if (PyDict_SetItemString(dict, "poses", poses) < 0) {
            Py_DECREF(poses);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(poses);
        return dict;
    }

    PyObject* frame_to_dict(const vp_sdk::vp_jpeg_frame& frame) {
        auto* dict = PyDict_New();
        if (dict == nullptr) {
            return nullptr;
        }
        if (set_item(dict, "channel_index", PyLong_FromLong(frame.channel_index)) < 0 ||
            set_item(dict, "frame_index", PyLong_FromLong(frame.frame_index)) < 0 ||
            set_item(dict, "width", PyLong_FromLong(frame.width)) < 0 ||
            set_item(dict, "height", PyLong_FromLong(frame.height)) < 0 ||
            set_item(dict, "jpeg", PyBytes_FromStringAndSize(reinterpret_cast<const char*>(frame.jpeg.data()), static_cast<Py_ssize_t>(frame.jpeg.size()))) < 0) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    }

    PyObject* Pipeline_new(PyTypeObject* type, PyObject*, PyObject*) {
        auto* self = reinterpret_cast<PipelineObject*>(type->tp_alloc(type, 0));
        if (self != nullptr) {
            self->pipeline = nullptr;
        }
        return reinterpret_cast<PyObject*>(self);
    }

    int Pipeline_init(PipelineObject* self, PyObject* args, PyObject* kwargs) {
        const char* yolo_engine = nullptr;
        const char* pose_engine = nullptr;
        const char* labels = nullptr;
        int enable_frame_output = 1;
        int enable_osd = 1;
        int result_queue_size = 100;
        int frame_queue_size = 3;
        int jpeg_quality = 82;
        int jpeg_max_width = 960;
        int device_id = 0;
        int skip_interval = 0;
        const char* decoder = "avdec_h264";
        double score_threshold = 0.35;
        double nms_threshold = 0.45;

        static char* kwlist[] = {
            const_cast<char*>("yolo_engine"),
            const_cast<char*>("pose_engine"),
            const_cast<char*>("labels"),
            const_cast<char*>("enable_frame_output"),
            const_cast<char*>("enable_osd"),
            const_cast<char*>("result_queue_size"),
            const_cast<char*>("frame_queue_size"),
            const_cast<char*>("jpeg_quality"),
            const_cast<char*>("jpeg_max_width"),
            const_cast<char*>("device_id"),
            const_cast<char*>("skip_interval"),
            const_cast<char*>("decoder"),
            const_cast<char*>("score_threshold"),
            const_cast<char*>("nms_threshold"),
            nullptr
        };

        if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "sss|ppiiiiiisdd",
                kwlist,
                &yolo_engine,
                &pose_engine,
                &labels,
                &enable_frame_output,
                &enable_osd,
                &result_queue_size,
                &frame_queue_size,
                &jpeg_quality,
                &jpeg_max_width,
                &device_id,
                &skip_interval,
                &decoder,
                &score_threshold,
                &nms_threshold)) {
            return -1;
        }

        vp_sdk::vp_pipeline_config config;
        config.yolo_engine = yolo_engine;
        config.pose_engine = pose_engine;
        config.labels = labels;
        config.enable_frame_output = enable_frame_output != 0;
        config.enable_osd = enable_osd != 0;
        config.result_queue_size = result_queue_size;
        config.frame_queue_size = frame_queue_size;
        config.jpeg_quality = jpeg_quality;
        config.jpeg_max_width = jpeg_max_width;
        config.device_id = device_id;
        config.skip_interval = skip_interval;
        config.decoder = decoder;
        config.score_threshold = static_cast<float>(score_threshold);
        config.nms_threshold = static_cast<float>(nms_threshold);

        try {
            self->pipeline = new vp_sdk::vp_pose_pipeline(config);
        }
        catch (const std::exception& exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
            return -1;
        }
        catch (const char* exc) {
            PyErr_SetString(PyExc_RuntimeError, exc);
            return -1;
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown non-standard exception");
            return -1;
        }
        return 0;
    }

    void Pipeline_dealloc(PipelineObject* self) {
        delete self->pipeline;
        self->pipeline = nullptr;
        Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
    }

    PyObject* Pipeline_start_file(PipelineObject* self, PyObject* args, PyObject* kwargs) {
        const char* path = nullptr;
        int cycle = 1;
        static char* kwlist[] = {const_cast<char*>("path"), const_cast<char*>("cycle"), nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|p", kwlist, &path, &cycle)) {
            return nullptr;
        }
        try {
            self->pipeline->start_file(path, cycle != 0);
        }
        catch (const std::exception& exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
            return nullptr;
        }
        catch (const char* exc) {
            PyErr_SetString(PyExc_RuntimeError, exc);
            return nullptr;
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown non-standard exception");
            return nullptr;
        }
        Py_RETURN_NONE;
    }

    PyObject* Pipeline_start_rtsp(PipelineObject* self, PyObject* args) {
        const char* url = nullptr;
        if (!PyArg_ParseTuple(args, "s", &url)) {
            return nullptr;
        }
        try {
            self->pipeline->start_rtsp(url);
        }
        catch (const std::exception& exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
            return nullptr;
        }
        catch (const char* exc) {
            PyErr_SetString(PyExc_RuntimeError, exc);
            return nullptr;
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown non-standard exception");
            return nullptr;
        }
        Py_RETURN_NONE;
    }

    PyObject* Pipeline_stop(PipelineObject* self, PyObject*) {
        self->pipeline->stop();
        Py_RETURN_NONE;
    }

    PyObject* Pipeline_read_result(PipelineObject* self, PyObject* args, PyObject* kwargs) {
        int timeout_ms = 0;
        static char* kwlist[] = {const_cast<char*>("timeout_ms"), nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &timeout_ms)) {
            return nullptr;
        }
        std::optional<vp_sdk::vp_pipeline_result> result;
        Py_BEGIN_ALLOW_THREADS
        result = self->pipeline->read_result(timeout_ms);
        Py_END_ALLOW_THREADS
        if (!result) {
            Py_RETURN_NONE;
        }
        return result_to_dict(*result);
    }

    PyObject* Pipeline_read_jpeg_frame(PipelineObject* self, PyObject* args, PyObject* kwargs) {
        int timeout_ms = 0;
        static char* kwlist[] = {const_cast<char*>("timeout_ms"), nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &timeout_ms)) {
            return nullptr;
        }
        std::optional<vp_sdk::vp_jpeg_frame> frame;
        Py_BEGIN_ALLOW_THREADS
        frame = self->pipeline->read_jpeg_frame(timeout_ms);
        Py_END_ALLOW_THREADS
        if (!frame) {
            Py_RETURN_NONE;
        }
        return frame_to_dict(*frame);
    }

    PyObject* Pipeline_is_running(PipelineObject* self, PyObject*) {
        if (self->pipeline->is_running()) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }

    PyObject* Pipeline_error(PipelineObject* self, PyObject*) {
        return PyUnicode_FromString(self->pipeline->error().c_str());
    }

    PyObject* Pipeline_stats(PipelineObject* self, PyObject*) {
        auto* dict = PyDict_New();
        if (dict == nullptr) {
            return nullptr;
        }
        if (set_item(dict, "dropped_results", PyLong_FromSize_t(self->pipeline->dropped_results())) < 0 ||
            set_item(dict, "dropped_frames", PyLong_FromSize_t(self->pipeline->dropped_frames())) < 0) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    }

    PyMethodDef Pipeline_methods[] = {
        {"start_file", reinterpret_cast<PyCFunction>(Pipeline_start_file), METH_VARARGS | METH_KEYWORDS, nullptr},
        {"start_rtsp", reinterpret_cast<PyCFunction>(Pipeline_start_rtsp), METH_VARARGS, nullptr},
        {"stop", reinterpret_cast<PyCFunction>(Pipeline_stop), METH_NOARGS, nullptr},
        {"read_result", reinterpret_cast<PyCFunction>(Pipeline_read_result), METH_VARARGS | METH_KEYWORDS, nullptr},
        {"read_jpeg_frame", reinterpret_cast<PyCFunction>(Pipeline_read_jpeg_frame), METH_VARARGS | METH_KEYWORDS, nullptr},
        {"is_running", reinterpret_cast<PyCFunction>(Pipeline_is_running), METH_NOARGS, nullptr},
        {"error", reinterpret_cast<PyCFunction>(Pipeline_error), METH_NOARGS, nullptr},
        {"stats", reinterpret_cast<PyCFunction>(Pipeline_stats), METH_NOARGS, nullptr},
        {nullptr, nullptr, 0, nullptr}
    };

    PyTypeObject PipelineType = {
        PyVarObject_HEAD_INIT(nullptr, 0)
    };

    PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "videopipe_sdk",
        nullptr,
        -1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };

}

PyMODINIT_FUNC PyInit_videopipe_sdk(void) {
    PipelineType.tp_name = "videopipe_sdk.Pipeline";
    PipelineType.tp_basicsize = sizeof(PipelineObject);
    PipelineType.tp_itemsize = 0;
    PipelineType.tp_flags = Py_TPFLAGS_DEFAULT;
    PipelineType.tp_new = Pipeline_new;
    PipelineType.tp_init = reinterpret_cast<initproc>(Pipeline_init);
    PipelineType.tp_dealloc = reinterpret_cast<destructor>(Pipeline_dealloc);
    PipelineType.tp_methods = Pipeline_methods;

    if (PyType_Ready(&PipelineType) < 0) {
        return nullptr;
    }

    auto* module = PyModule_Create(&moduledef);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&PipelineType);
    if (PyModule_AddObject(module, "Pipeline", reinterpret_cast<PyObject*>(&PipelineType)) < 0) {
        Py_DECREF(&PipelineType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
