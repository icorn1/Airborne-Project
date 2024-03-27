package com.arbrn.urcap.demonstration.impl;

import com.ur.urcap.api.contribution.installation.swing.SwingInstallationNodeView;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

public class TestInstallationNodeView implements SwingInstallationNodeView<TestInstallationNodeContribution> {

    private Option ACTION_TYPE = Option.Run;
    private Laminate SELECTED_LAMINATE = Laminate.Laminate_1;

    private JLabel RETURN_VALUE = new JLabel();

    private enum Option {
        Run("move_ply(", "Run"),
        Calibrate("calibration(", "Calibrate");

        private final String id;
        private final String displayName;

        Option(String id, String displayName) {
            this.id = id;
            this.displayName = displayName;
        }

        public String getId() {
            return id;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

	private enum Laminate {
		Laminate_1("1", "Laminate 1"),
		Laminate_2("2", "Laminate 2"),
		Laminate_3("3", "Laminate 3");

        private final String id;
        private final String displayName;

        Laminate(String id, String displayName) {
            this.id = id;
            this.displayName = displayName;
        }

        public String getId() {
            return id;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

	@Override
	public void buildUI(JPanel panel, TestInstallationNodeContribution contribution) {
		panel.setLayout(new BorderLayout());

        // Top right logo
        JPanel logoPanel = new JPanel(new BorderLayout());
        logoPanel.add(createLogoPanel(), BorderLayout.NORTH);
        panel.add(logoPanel, BorderLayout.NORTH);
		panel.add(createSpacer(40));

		// Support image below the logo
		// Add the image panel below the logo panel
		panel.add(addSupportImageUR("/com/arbrn/urcap/demonstration/logo.png"));							 

		// Big green button "Start" below the text
		panel.add(createSenderTestButton(contribution), BorderLayout.CENTER);

		// Settings space below the button, bottom left
		JPanel settingsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT)); // Use FlowLayout to left-align components
		settingsPanel.add(createActionToPerform());
		settingsPanel.add(createLaminateSelection());
		panel.add(settingsPanel, BorderLayout.SOUTH);
	}

	private JPanel createLogoPanel() {
		JPanel logoPanel = new JPanel(new BorderLayout());
		JLabel logoLabel = new JLabel();
		ImageIcon logoIcon = new ImageIcon(getClass().getResource("/com/arbrn/urcap/demonstration/logo.png"));
		logoLabel.setIcon(new ImageIcon(logoIcon.getImage().getScaledInstance(250, 144, Image.SCALE_SMOOTH)));
		logoPanel.add(logoLabel, BorderLayout.EAST); // Align the logo to the right
		return logoPanel;
	}

	private JPanel addSupportImageUR(String imagePath) {
		// Load the image
		ImageIcon imageIcon = new ImageIcon(getClass().getResource(imagePath));

		// Resize the image
		int targetWidth = 150; // Adjust the width as needed
		int targetHeight = (int) ((double) imageIcon.getIconHeight() / imageIcon.getIconWidth() * targetWidth);
		Image scaledImage = imageIcon.getImage().getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
		imageIcon = new ImageIcon(scaledImage);

		// Create and configure the image label
		JLabel imageLabel = new JLabel(imageIcon);
		// imageLabel.setHorizontalAlignment(SwingConstants.CENTER); // Center the image horizontally

		// Create a panel to hold the image
		JPanel imagePanel = new JPanel(new BorderLayout());
		imagePanel.add(Box.createVerticalStrut(20), BorderLayout.NORTH); // Add spacing
		imagePanel.add(imageLabel, BorderLayout.CENTER); // Add the image label to the center

		return imagePanel;
	}


	private Box createSenderTestButton(final TestInstallationNodeContribution contribution) {
		Box box = Box.createVerticalBox();

		JLabel labelText = new JLabel("Press the following button to start the program once you have the right configuration.");
		labelText.setAlignmentX(Component.CENTER_ALIGNMENT);
		labelText.setFont(new Font("Arial", Font.PLAIN, 20)); // Increase font size
		box.add(labelText);

		JButton button = new JButton("START");
		button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				contribution.sendScriptTest(ACTION_TYPE.getId(), SELECTED_LAMINATE.getId());
			}
		});
		button.setBackground(new Color(24, 196, 12));
		button.setForeground(Color.WHITE);
		Dimension buttonSize = new Dimension(250, 70); // Increase button size
		button.setMaximumSize(buttonSize);
		button.setMinimumSize(buttonSize);
		button.setPreferredSize(buttonSize);
		button.setAlignmentX(Component.CENTER_ALIGNMENT); // Center the button horizontally
		button.setFont(new Font("Arial", Font.BOLD, 30)); // Increase button font size
		box.add(createSpacer(20)); // Add space between text and button
		box.add(button);

		return box;
	}


    private JComboBox<Option> createActionToPerform() {
        final JComboBox<Option> jcb = new JComboBox<Option>(new DefaultComboBoxModel<Option>(Option.values()));
        jcb.setSelectedIndex(0);
        jcb.setPreferredSize(new Dimension(150, 30));
        jcb.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED) {
                    Option selectedOption = (Option) jcb.getSelectedItem();
                    System.out.println("Selected option: " + selectedOption.getDisplayName());
                    ACTION_TYPE = selectedOption;
                }
            }
        });
        return jcb;
    }

	public void setReturnValue(String value) {
		RETURN_VALUE.setText(value);
	}

    private JComboBox<Laminate> createLaminateSelection() {
        final JComboBox<Laminate> jcb = new JComboBox<Laminate>(Laminate.values());
        jcb.setSelectedItem(SELECTED_LAMINATE);
        jcb.setPreferredSize(new Dimension(150, 30));
        jcb.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED) {
                    SELECTED_LAMINATE = (Laminate) jcb.getSelectedItem();
                    System.out.println("Selected option: " + SELECTED_LAMINATE.getDisplayName());
                }
            }
        });
        return jcb;
    }


    private Component createSpacer(int height) {
        return Box.createRigidArea(new Dimension(0, height));
    }
}
